"""Consistency checks of the generation model against recovered methane.

The model is compared, realization by realization, with every independent record of methane recovered at a Colombian disposal site (inputs/records/):

* the metered flow of the Doña Juana collection field, 2010-2025, read at 55 % and at 50 % methane, as in the Methods;
* the methane destroyed that the Clean Development Mechanism monitoring reports claim at Antanas (2009-2011), El Guacal (2011-2013) and Presidente (2020);
* the flare flows that the SSPD inspections of 2019 recorded at La Pradera and Presidente;
* the flare flows that operators reported to the SUI register at Loma de los Cocos (2016), El Carrasco (2014), Guayabal (2014) and La Cortada (2016).

Recovered methane cannot exceed what is generated, so each record gives the collection efficiency that the model implies, recovered over modeled generation for the same period. A record whose implied efficiency exceeds one is not compatible with the model; a set of implied efficiencies well below the field values of Table 7 says that either the fields underperform or the generation is overestimated, and the check cannot tell which.

The generation is recomputed with the same realizations as the base case (same seed and parameter stream), extended back to 2005 for the selected sites.

Writes
    results/consistency_dj_annual.csv
    results/consistency_sites.csv
    results/consistency_gen_draws.npz
Run from the project root:
    python scripts/14_consistency.py [NDRAW]
"""
import datetime as dt
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
NDRAW = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10000)
sys.argv = [sys.argv[0], str(NDRAW)]
M = importlib.import_module("11_lfg_h2_model")

OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
MA = ROOT / "inputs" / "records"

YX = np.arange(2005, 2026)                     # evaluated years of the check
AGEX = YX[None, :] - M.DEP[:, None]
RHO_N = 0.7168                                 # kg CH4 per normal m3 (0 C), as the operator's record
HOURS = 8760.0

SITES = {
    "11001|DONA JUANA": ("Doña Juana", "Bogotá"),
    "52001|ANTANAS": ("Antanas", "Pasto"),
    "05347|EL GUACAL": ("El Guacal", "Envigado"),
    "76670|PRESIDENTE": ("Presidente", "Palmira"),
    "05237|LA PRADERA": ("La Pradera", "Medellín"),
    "13838|LOMA DE LOS COCOS": ("Loma de los Cocos", "Cartagena"),
    "68001|EL CARRASCO": ("El Carrasco", "Bucaramanga"),
    "54001|GUAYABAL": ("Guayabal", "Cúcuta"),
    "54518|LA CORTADA": ("La Cortada", "Pamplona"),
}


def kernel(k):
    """decay_kernel of the model on the extended evaluation years."""
    tau = AGEX[:, :, None] + 0.5 - M.SEC_OFF[None, None, :] - M.LAG
    return np.where(tau >= 0.0, np.exp(-k * np.clip(tau, 0.0, None)), 0.0).mean(axis=2)


def generation(p, idx):
    """Methane generated at the sites idx, m3/yr, on the years YX, for one realization."""
    kz = np.empty((4, 5))
    for c in range(4):
        for z in range(5):
            kz[c, z] = M.tri_ppf(p["u_k"][c], M.K[c, min(z + 1, 4)], M.K[c, z], M.K[c, max(z - 1, 0)])
    l0 = M.L0C * p["l0m"]
    tot = np.maximum(M.W[idx].sum(axis=1), 1e-9)
    wbar = np.clip(M.Wc[idx].sum(axis=2) / tot[:, None], 1e-6, None)
    rest = np.clip(1.0 - wbar.sum(axis=1, keepdims=True), 1e-6, None)
    sd = np.where(M.share_measured[idx][:, None] > 0.5, 0.10, 0.20)
    alr = np.log(wbar / rest) + sd * p["z_comp"][idx]
    e = np.exp(alr)
    wnew = e / (1.0 + e.sum(axis=1, keepdims=True))
    rmul = wnew / wbar
    G = np.zeros((len(idx), len(YX)))
    zs = M.zone[idx]
    for j, s in enumerate(idx):
        for c in range(4):
            k = kz[c, zs[j]]
            G[j] += k * l0[c] * rmul[j, c] * (M.Wc[s, c, :] @ kernel(k))
    return G * M.mcf0[idx][:, None]


def days(a, b):
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1


def evidence():
    """Records of recovered methane, t CH4 per year, with their period."""
    rec = []
    # Doña Juana, metered flow of the operator, annual means
    dj = pd.read_csv(MA / "dona_juana_biogas_year.csv")
    for _, r in dj.iterrows():
        y = int(r["year"])
        t55 = r["flow_m3n_h"] * HOURS * 0.55 * RHO_N / 1000.0
        t50 = r["flow_m3n_h"] * HOURS * 0.50 * RHO_N / 1000.0
        rec.append(dict(site="11001|DONA JUANA", start=f"{y}-01-01", end=f"{y}-12-31", t_yr=t55,
                        t_yr_lo=t50, t_yr_hi=t55, basis="metered flow (operator)",
                        source="cgrdj2026biogas"))
    # Clean Development Mechanism monitoring reports, methane destroyed claimed, by period
    cdm = [("52001|ANTANAS", "2009-11-03", "2009-12-31", 229.24),
           ("52001|ANTANAS", "2010-01-01", "2010-12-31", 1160.578),
           ("52001|ANTANAS", "2011-01-01", "2011-11-09", 843.164),
           ("05347|EL GUACAL", "2011-02-23", "2011-12-31", 1830.429),
           ("05347|EL GUACAL", "2012-01-01", "2012-12-31", 4154.603),
           ("05347|EL GUACAL", "2013-01-01", "2013-10-31", 3831.016),
           ("76670|PRESIDENTE", "2020-06-01", "2020-12-31", 516.0)]
    for s, a, b, t in cdm:
        f = 365.0 / days(a, b)
        rec.append(dict(site=s, start=a, end=b, t_yr=t * f, t_yr_lo=t * f, t_yr_hi=t * f,
                        basis="CDM monitoring report", source="cdm"))
    # SSPD inspections of 2019, flare flows (inputs/site_facts.csv)
    rec.append(dict(site="05237|LA PRADERA", start="2019-01-01", end="2019-12-31", t_yr=3799.0,
                    t_yr_lo=3140.0, t_yr_hi=4144.0, basis="flare flow, SSPD inspection",
                    source="sspd2019pradera"))
    rec.append(dict(site="76670|PRESIDENTE", start="2019-01-01", end="2019-12-31", t_yr=1554.0,
                    t_yr_lo=1209.0, t_yr_hi=1899.0, basis="flare flow, SSPD inspection",
                    source="sspd2019presidente"))
    # flare flows reported to the SUI register, read as continuous flows at 15 C
    sui = pd.read_csv(MA / "sui_gas_systems.csv")
    for s in ("13838|LOMA DE LOS COCOS", "68001|EL CARRASCO", "54001|GUAYABAL", "54518|LA CORTADA"):
        key = {"13838|LOMA DE LOS COCOS": "13838|RELLENO SANITARIO PARQUE AMBIENTAL LOMA DE LOS COCOS",
               "68001|EL CARRASCO": "68001|RELLENO SANITARIO EL CARRASCO",
               "54001|GUAYABAL": "54001|RELLENO SANITARIO GUAYABAL",
               "54518|LA CORTADA": "54518|RELLENO SANITARIO LA CORTADA"}[s]
        r = sui[sui["site_id"] == key].iloc[0]
        y = int(r["last_report_year"])
        t = r["flare_ch4_m3_h"] * HOURS * M.RHO_CH4 / 1000.0
        rec.append(dict(site=s, start=f"{y}-01-01", end=f"{y}-12-31", t_yr=t, t_yr_lo=t, t_yr_hi=t,
                        basis="flare flow, SUI register", source="sspd2026sui",
                        lfg_m3_h=float(r["flare_flow_m3_h"]), ch4_pct=float(r["flare_ch4_pct"]),
                        wells=float(r["gas_wells"]), tpd=float(r["tpd_last_year"])))
    return pd.DataFrame(rec)


def period_mean(G_row, a, b):
    """Mean generation rate over a dated period, weighting each calendar year by its days."""
    y0, y1 = int(a[:4]), int(b[:4])
    w = []
    for y in range(y0, y1 + 1):
        d0 = max(dt.date.fromisoformat(a), dt.date(y, 1, 1))
        d1 = min(dt.date.fromisoformat(b), dt.date(y, 12, 31))
        w.append(((d1 - d0).days + 1, y))
    tot = sum(x for x, _ in w)
    return sum(x * G_row[..., int(np.searchsorted(YX, y))] for x, y in w) / tot


def main():
    idx = np.array([M.sidx[s] for s in SITES])
    Gd = np.empty((NDRAW, len(idx), len(YX)), dtype=np.float32)
    for d in range(NDRAW):
        p = M.draw_params()
        Gd[d] = generation(p, idx)
        if (d + 1) % 2000 == 0:
            print(f"  realization {d + 1}/{NDRAW}", flush=True)
    np.savez_compressed(OUT / "consistency_gen_draws.npz", sites=np.array(list(SITES)), years=YX,
                        G=Gd)
    ev = evidence()
    rows = []
    for _, r in ev.iterrows():
        j = list(SITES).index(r["site"])
        g = period_mean(Gd[:, j, :].astype(float), r["start"], r["end"]) * M.RHO_CH4 / 1000.0  # t/yr
        eta = r["t_yr"] / g
        eta_lo = r["t_yr_lo"] / g
        row = dict(site=r["site"], name=SITES[r["site"]][0], city=SITES[r["site"]][1],
                   start=r["start"], end=r["end"], basis=r["basis"], source=r["source"],
                   rec_t_yr=r["t_yr"], rec_t_yr_lo=r["t_yr_lo"], rec_t_yr_hi=r["t_yr_hi"],
                   gen_t_p10=np.percentile(g, 10), gen_t_p50=np.percentile(g, 50),
                   gen_t_p90=np.percentile(g, 90),
                   eta_p10=np.percentile(eta, 10), eta_p50=np.percentile(eta, 50),
                   eta_p90=np.percentile(eta, 90), eta_lo_p50=np.percentile(eta_lo, 50),
                   share_eta_gt1=float((eta > 1).mean()))
        for k in ("lfg_m3_h", "ch4_pct", "wells", "tpd"):
            if k in r and pd.notna(r.get(k)):
                row[k] = r[k]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "consistency_sites.csv", index=False, float_format="%.4g")
    dj = out[out["site"] == "11001|DONA JUANA"].copy()
    dj["year"] = dj["start"].str[:4].astype(int)
    dj[["year", "rec_t_yr", "rec_t_yr_lo", "gen_t_p10", "gen_t_p50", "gen_t_p90", "eta_p10",
        "eta_p50", "eta_p90", "eta_lo_p50"]].to_csv(OUT / "consistency_dj_annual.csv", index=False,
                                                     float_format="%.4g")
    pd.set_option("display.width", 220)
    print(out.drop(columns=["source"]).round(3).to_string())


if __name__ == "__main__":
    main()
