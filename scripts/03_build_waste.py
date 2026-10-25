"""Waste deposited at each canonical disposal site, 1985-2045.

Observed years
    2010-2024 from the disposal registry, allocated to canonical sites by scripts/02_site_catalog.py. Every delivery of a served municipality is kept, including municipalities that split their waste between two or more sites in the same year.

Gaps inside the observed period
    A municipality-year missing between two observed years (mostly 2015-2017) takes the linear interpolation of the municipality's own tonnage and the site shares of the nearest observed year (the earlier one on a tie).

Back-cast, before the municipality's first observed year
    Constant disposal per urban inhabitant at the level of the first three observed years, with the site shares of the first observed year. A site receives back-cast waste only from its back-cast start, which is its documented opening year (inputs/site_facts.csv, the SUI register, or the first registry year when the site appears after 2010), cut at 1985, or 1995 when the opening is not documented.
    Waste of 1995-2009 that falls before a documented opening is kept in a pre-opening pool, reported but assigned to no site.

Projection, after 2024
    Constant disposal per urban inhabitant at the 2020-2024 level of each municipality, with its 2024 site shares. Population follows the DANE projection to 2042 and is held at its 2042 value afterwards.

Licence layer
    For each active site the projected deposits are split at the licensed end of life, never earlier than 1 January 2026, into a licensed layer and a beyond-licence layer. An expired licence is cut at 1 January 2026. A site with no reported date takes the median remaining life of the active sites of its type that report one. The base case keeps both layers (deposits continue in place); the licensed-only variant drops the second.

Outputs
    results/waste_municipality_site_year.csv  served municipality x site x year
    results/waste_site_year.csv               site x year, with the licence layer
    results/waste_pools.csv                   pre-opening and unallocated pools
    results/waste_checks.json                 totals quoted in the text

Run from the project root after scripts/01_pop.py and scripts/02_site_catalog.py:
    python scripts/03_build_waste.py
"""

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

Y_DOC0 = 1985          # earliest back-cast year for a documented opening
Y_DEF0 = 1995          # back-cast start when the opening is not documented
Y_END = 2045
OBS0, OBS1 = 2010, 2024
PROJ_WINDOW = (2020, 2024)
HORIZON_START = pd.Timestamp("2026-01-01")
POP_LAST = 2042


def population():
    pop = pd.read_csv(RESULTS / "population_municipal.csv", dtype={"municipality_code": str})
    pop["municipality_code"] = pop["municipality_code"].str.zfill(5)
    p = pop.pivot_table(index="municipality_code", columns="year", values="pop_urban")
    t = pop.pivot_table(index="municipality_code", columns="year", values="pop_total")
    p = p.where(p > 0)
    # a municipality with no urban count uses its total; years without a count take the nearest counted year (a municipality created in 2022 is counted from 2024)
    none = ~p.notna().any(axis=1)
    p.loc[none] = t.loc[none].where(t.loc[none] > 0)
    p = p.ffill(axis=1).bfill(axis=1).fillna(1.0)
    g = ((p[2012] / p[2005]) ** (1.0 / 7.0)).clip(0.97, 1.05)
    for y in range(Y_DOC0, 2005):
        p[y] = p[2005] * g ** (y - 2005)
    for y in range(POP_LAST + 1, Y_END + 1):
        p[y] = p[POP_LAST]
    return p[sorted(p.columns)]


def backcast_start(cat):
    doc = cat["opened_source"].fillna("not documented") != "not documented"
    start = np.where(doc, np.maximum(cat["opened_year"].fillna(Y_DEF0), Y_DOC0), Y_DEF0)
    return pd.Series(start.astype(int), index=cat.index), pd.Series(doc.values, index=cat.index)


def decimal_year(ts):
    ts = pd.to_datetime(ts)
    start = pd.to_datetime(ts.dt.year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    days = np.where(ts.dt.is_leap_year, 366.0, 365.0)
    return ts.dt.year + (ts - start).dt.days / days


def licence_cut(cat):
    """Decimal year after which projected deposits count as beyond the licence."""
    act = cat[cat["status_2024"] == "active"].copy()
    h = HORIZON_START.year + 0.0
    end = decimal_year(act["licence_end"])
    cut = end.where(end > h, h)
    life = act["remaining_life_2026"].clip(lower=0.0)
    med = life.groupby(act["site_type"]).median()
    miss = end.isna()
    add = act.loc[miss, "site_type"].map(med).fillna(life.median())
    cut.loc[miss] = h + add
    return cut


def main():
    cat = pd.read_csv(RESULTS / "site_catalog.csv", index_col=0)
    rows = pd.read_csv(RESULTS / "site_rows_assigned.csv", dtype={"served_dane_code": str})
    rows["served_dane_code"] = rows["served_dane_code"].str.zfill(5)
    pop = population()
    start, documented = backcast_start(cat)
    active = set(cat.index[cat["status_2024"] == "active"])
    first_year, last_year = cat["first_year"].to_dict(), cat["last_year"].to_dict()

    obs = rows.groupby(["served_dane_code", "year", "site_id"])["tons"].sum()
    out, pools = [], []
    missing_pop = []
    for m, d in obs.groupby(level=0):
        d = d.droplevel(0)
        tot = d.groupby(level=0).sum()
        yrs = np.array(sorted(tot.index))
        if m not in pop.index:
            missing_pop.append(m)
            for (y, s), t in d.items():
                out.append((m, s, y, t, "observed"))
            continue
        P = pop.loc[m]
        first = yrs[:3]
        pc_first = tot[first].sum() / P[first].sum()
        win = yrs[(yrs >= PROJ_WINDOW[0]) & (yrs <= PROJ_WINDOW[1])]
        win = win if len(win) else yrs[-3:]
        pc_last = tot[win].sum() / P[win].sum()
        sh = {y: d.loc[y] / d.loc[y].sum() for y in yrs}
        y0, y1 = yrs[0], yrs[-1]
        for y in range(Y_DOC0, Y_END + 1):
            if y in sh:
                for s, t in d.loc[y].items():
                    out.append((m, s, y, t, "observed"))
            elif y < y0:
                T = pc_first * P[y]
                pre = 0.0
                for s, w in sh[y0].items():
                    if y >= start[s]:
                        out.append((m, s, y, T * w, "backcast"))
                    elif documented[s] and y >= Y_DEF0:
                        pre += T * w
                if pre > 0:
                    pools.append((m, y, "pre_opening", pre))
            elif y > y1:
                T = pc_last * P[y]
                w = sh[y1][[s for s in sh[y1].index if s in active]]
                if w.sum() <= 0:
                    pools.append((m, y, "unallocated", T))
                    continue
                w = w / w.sum()
                for s, v in w.items():
                    out.append((m, s, y, T * v, "projection"))
            else:
                lo, hi = yrs[yrs < y].max(), yrs[yrs > y].min()
                T = tot[lo] + (tot[hi] - tot[lo]) * (y - lo) / (hi - lo)
                order = (lo, hi) if (y - lo) <= (hi - y) else (hi, lo)
                use = sh[order[0]]
                for cand in order:
                    ok = [s for s in sh[cand].index
                          if first_year[s] <= y <= last_year[s]]
                    if ok:
                        use = sh[cand][ok] / sh[cand][ok].sum()
                        break
                for s, v in use.items():
                    out.append((m, s, y, T * v, "gap"))

    w = pd.DataFrame(out, columns=["served_dane_code", "site_id", "year", "tons", "kind"])
    w.to_csv(RESULTS / "waste_municipality_site_year.csv", index=False, float_format="%.3f")
    pl = pd.DataFrame(pools, columns=["served_dane_code", "year", "pool", "tons"])

    site = w.pivot_table(index=["site_id", "year"], columns="kind", values="tons",
                         aggfunc="sum", fill_value=0.0)
    site["tons"] = site.sum(axis=1)
    site = site.reset_index()
    cut = licence_cut(cat)
    c = site["site_id"].map(cut)
    frac = ((site["year"] + 1.0) - c).clip(0.0, 1.0).fillna(0.0)
    site["tons_beyond_licence"] = site["tons"] * frac
    site["licence_cut"] = c.round(3)
    cols = ["site_id", "year", "tons", "tons_beyond_licence"] + \
           [k for k in ("observed", "gap", "backcast", "projection") if k in site.columns] + \
           ["licence_cut"]
    site[cols].to_csv(RESULTS / "waste_site_year.csv", index=False, float_format="%.3f")

    pool = pl.pivot_table(index="year", columns="pool", values="tons", aggfunc="sum",
                          fill_value=0.0)
    pool.to_csv(RESULTS / "waste_pools.csv", float_format="%.1f")

    nat = site.groupby("year")["tons"].sum()
    reg = rows.groupby("year")["tons"].sum()
    chk = {
        "municipalities": int(obs.index.get_level_values(0).nunique()),
        "municipalities_without_population": missing_pop,
        "sites_with_deposits": int(site["site_id"].nunique()),
        "registry_total_Mt": {int(y): round(v / 1e6, 3) for y, v in reg.items()},
        "series_total_Mt": {int(y): round(v / 1e6, 3) for y, v in nat.items()
                            if y in (1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020,
                                     2024, 2025, 2026, 2030, 2035, 2040, 2045)},
        "gap_fill_Mt_2010_2024": round(float(w.loc[w.kind == "gap", "tons"].sum()) / 1e6, 3),
        "pre_opening_pool_Mt_1995_2009": round(float(pl.loc[pl.pool == "pre_opening", "tons"].sum()) / 1e6, 3),
        "unallocated_pool_Mt": round(float(pl.loc[pl.pool == "unallocated", "tons"].sum()) / 1e6, 3),
        "beyond_licence_share_2026_2040": round(float(
            site.loc[site.year.between(2026, 2040), "tons_beyond_licence"].sum()
            / site.loc[site.year.between(2026, 2040), "tons"].sum()), 4),
    }
    obs_site = w[w.kind.isin(["observed", "gap"])].groupby("site_id")["year"].agg(["min", "max"])
    mism = (obs_site["max"] != cat.loc[obs_site.index, "last_year"]).sum()
    chk["sites_whose_last_deposit_differs_from_catalogue"] = int(mism)
    (RESULTS / "waste_checks.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps({k: v for k, v in chk.items() if k != "registry_total_Mt"}, indent=1))


if __name__ == "__main__":
    main()
