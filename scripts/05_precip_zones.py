"""
Precipitation-zone assignment for the disposal sites of the landfill-gas hydrogen atlas.

Each site takes the zone of its mean annual precipitation, derived from the published thirty-year climatological normals of IDEAM, in place of the fallback zoning of lfg_h2_model.py (moderately wet everywhere except four anchored cities).

Source: Normales Climatologicas de Colombia, Datos Abiertos Colombia, dataset nsz2-kzcq, read from inputs/ideam_normales_nsz2-kzcq.csv (about 16,800 rows covering four normal periods and eight parameters).

Outputs
    results/site_precipitation.csv   one row per site, normal and zone
    results/precip_stats.json        the robustness statistics
    results/numbers_precip.csv       the values quoted in the Methods section

Run from the project root:
    python scripts/05_precip_zones.py

The model reads results/site_precipitation.csv by site_id (zone_9120).
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"

API = "https://www.datos.gov.co/resource/nsz2-kzcq.json"
APP_TOKEN = None          # optional; register one at dev.socrata.com to lift the anonymous quota
CACHE = INPUTS / "ideam_normales_nsz2-kzcq.csv"

DIVIPOLA = INPUTS / "DIVIPOLA_-_Códigos_cabeceras_-_Centros_poblados_20260718.xlsx"
SITES = RESULTS / "site_catalog.csv"

RADII_KM = (25.0, 50.0, 100.0, 200.0)   # widened in order until a station is found
IDW_POWER = 2.0
MAX_STATIONS = 8

# Colombia Landfill Gas Model zones, wettest first, to match the column order of K
ZONE_EDGES = (2000.0, 1500.0, 1000.0, 500.0)
ZONE_LABEL = {1: "very wet", 2: "wet", 3: "moderately wet", 4: "moderately dry", 5: "dry"}


def fetch_normals(force=False):
    """Download the whole dataset once and cache it next to the other inputs."""
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE)
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    r = requests.get(API, params={"$limit": 20000}, headers=headers, timeout=180)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    if d.empty:
        raise SystemExit("the download returned no rows; check the endpoint and any quota")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False, encoding="utf8")
    print(f"downloaded {len(d):,} rows to {CACHE}")
    return d


def stations(raw, periodo):
    """Precipitation normals for one period, cleaned and de-duplicated."""
    d = raw.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    par = d["par_metro"].astype(str).str.upper().str.strip()
    keep = par.str.startswith("PRECIPITAC") & (d["periodo"].astype(str).str.strip() == periodo)
    d = d.loc[keep, ["c_digo", "estaci_n", "municipio", "departamento",
                     "altitud_m", "longitud", "latitud", "anual"]].copy()
    for c in ("altitud_m", "longitud", "latitud", "anual"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["longitud", "latitud", "anual"])
    d = d[(d.anual > 0) & (d.longitud.between(-82, -66)) & (d.latitud.between(-5, 14))]
    d["c_digo"] = d["c_digo"].astype("Int64").astype(str).str.zfill(10)
    return d.drop_duplicates("c_digo").reset_index(drop=True)


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance, vectorised over the second pair."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def zone_of(mm):
    """Bin a mean annual precipitation onto the five zones of the model."""
    if not np.isfinite(mm):
        return np.nan
    for i, edge in enumerate(ZONE_EDGES, start=1):
        if mm >= edge:
            return i
    return 5


def assign(sites, st):
    """Inverse-distance-weighted normal at each site, widening the radius as needed."""
    out = []
    slon, slat = st.longitud.to_numpy(), st.latitud.to_numpy()
    sann = st.anual.to_numpy()
    for row in sites.itertuples(index=False):
        if not (np.isfinite(row.lon) and np.isfinite(row.lat)):
            out.append((np.nan, np.nan, 0, np.nan, np.nan))
            continue
        dist = haversine_km(row.lon, row.lat, slon, slat)
        order = np.argsort(dist)
        used_r = np.nan
        sel = np.array([], dtype=int)
        for r in RADII_KM:
            sel = order[dist[order] <= r][:MAX_STATIONS]
            if sel.size:
                used_r = r
                break
        if not sel.size:
            out.append((np.nan, np.nan, 0, np.nan, dist[order[0]]))
            continue
        d = np.maximum(dist[sel], 0.5)
        w = 1.0 / d ** IDW_POWER
        mm = float(np.sum(w * sann[sel]) / np.sum(w))
        out.append((mm, float(sann[order[0]]), int(sel.size), used_r, float(dist[order[0]])))
    cols = ["mm", "mm_nearest", "n_stations", "radius_km", "dist_nearest_km"]
    return pd.DataFrame(out, columns=cols, index=sites.index)


def main():
    raw = fetch_normals(force="--refresh" in sys.argv)
    st9120 = stations(raw, "1991-2020")
    st8110 = stations(raw, "1981-2010")
    if st9120.empty:
        raise SystemExit("no 1991-2020 precipitation normals found; check the download")

    dv = pd.read_excel(DIVIPOLA)
    dv["dane"] = dv["Codigo Municipio"].astype(int).astype(str).str.zfill(5)
    dv["lon"] = pd.to_numeric(dv["longitud"], errors="coerce")
    dv["lat"] = pd.to_numeric(dv["Latitud"], errors="coerce")
    dv = dv.dropna(subset=["lon", "lat"])
    seat = dv[dv["Tipo*"].astype(str).str.strip() == "CM"].drop_duplicates("dane")
    seat = seat.set_index("dane")[["lon", "lat"]]
    # municipalities absent from the seat list fall back to the mean of their populated centres
    other = dv.groupby("dane")[["lon", "lat"]].mean()
    dv = seat.combine_first(other).reset_index()

    sites = pd.read_csv(SITES, dtype={"dane": str}).reset_index(drop=True)
    sites["dane"] = sites["dane"].str.zfill(5)
    sites = sites.merge(dv, on="dane", how="left")

    a = assign(sites, st9120)
    b = assign(sites, st8110)

    res = pd.DataFrame({
        "site_id": sites.site_id,
        "display_name": sites.display_name,
        "department": sites.dept,
        "municipality": sites.municipality,
        "dane": sites.dane,
        "lon": sites.lon,
        "lat": sites.lat,
        "tons_2010_2024": sites.tons_2010_2024,
        "status_2024": sites.status_2024,
        "map_mm_9120": a.mm.round(1),
        "n_stations_9120": a.n_stations,
        "radius_km_9120": a.radius_km,
        "dist_nearest_km": a.dist_nearest_km.round(1),
        "mm_nearest_9120": a.mm_nearest.round(1),
        "map_mm_8110": b.mm.round(1),
    })
    res["zone_9120"] = res.map_mm_9120.map(zone_of)
    res["zone_8110"] = res.map_mm_8110.map(zone_of)
    res["zone_nearest"] = res.mm_nearest_9120.map(zone_of)
    res["zone_label"] = res.zone_9120.map(lambda z: ZONE_LABEL.get(z, ""))

    ok_all = res.zone_9120.notna()
    # the article covers the sites active in 2024, so the checks of the assignment are reported over them; the registry-wide values are kept in the json with suffix _all
    ok = ok_all & (res.status_2024 == "active")
    w = res.tons_2010_2024.fillna(0.0)

    def share(mask, weights=None, base=None):
        base = ok if base is None else base
        m = mask & base
        if weights is None:
            return 100.0 * m.sum() / max(int(base.sum()), 1)
        return 100.0 * float(weights[m].sum()) / max(float(weights[base].sum()), 1e-9)

    stats = {
        "n_stations_9120": int(len(st9120)),
        "n_stations_8110": int(len(st8110)),
        "n_sites": int(len(res)),
        "n_sites_assigned": int(ok_all.sum()),
        "n_active_assigned": int(ok.sum()),
        "agree_period_pct_all": float(share(res.zone_9120 == res.zone_8110, base=ok_all)),
        "agree_period_pct_weighted_all": float(share(res.zone_9120 == res.zone_8110, w, base=ok_all)),
        "agree_nearest_pct_all": float(share(res.zone_9120 == res.zone_nearest, base=ok_all)),
        "agree_nearest_pct_weighted_all": float(share(res.zone_9120 == res.zone_nearest, w, base=ok_all)),
        "median_dist_nearest_km": float(res.loc[ok, "dist_nearest_km"].median()),
        "share_sites_within_25km": float(share(res.radius_km_9120 == 25.0)),
        "agree_period_pct": float(share(res.zone_9120 == res.zone_8110)),
        "agree_period_pct_weighted": float(share(res.zone_9120 == res.zone_8110, w)),
        "agree_nearest_pct": float(share(res.zone_9120 == res.zone_nearest)),
        "agree_nearest_pct_weighted": float(share(res.zone_9120 == res.zone_nearest, w)),
        "zone_counts_9120": {int(k): int(v) for k, v in res.zone_9120.value_counts().items()},
        "zone_tonnage_share_9120": {int(k): round(100.0 * float(w[res.zone_9120 == k].sum())
                                                  / float(w[ok].sum()), 1)
                                    for k in sorted(res.zone_9120.dropna().unique())},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "site_precipitation.csv", index=False)
    (RESULTS / "precip_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf8")

    numbers = (
        "name,value\n"
        f"PrecipNStations,{stats['n_stations_9120']}\n"
        f"PrecipAgreePeriod,{stats['agree_period_pct']:.0f}\n"
        f"PrecipAgreeNearest,{stats['agree_nearest_pct']:.0f}\n"
        f"PrecipAgreePeriodW,{stats['agree_period_pct_weighted']:.0f}\n"
        f"PrecipAgreeNearestW,{stats['agree_nearest_pct_weighted']:.0f}\n"
        f"PrecipWithinRadius,{stats['share_sites_within_25km']:.0f}\n"
    )
    (RESULTS / "numbers_precip.csv").write_text(numbers, encoding="utf8")

    print(json.dumps(stats, indent=1))
    missing = res.loc[~ok, ["site_id", "department", "municipality", "dist_nearest_km"]]
    if len(missing):
        print(f"\n{len(missing)} sites without an assignment, resolve these by hand\n")
        print(missing.to_string(index=False))
    cols = ["display_name", "municipality", "map_mm_9120", "zone_9120",
            "dist_nearest_km", "n_stations_9120"]
    print("\nTwelve largest sites by tonnage 2010-2024\n")
    print(res.nlargest(12, "tons_2010_2024")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
