"""Checks of the model against independent evidence.

1. Doña Juana, the site with the longest public metered record of collected gas. The metered methane destroyed in 2021-2025, from the operator's annual mean flow at 55 % (or 50 %) methane, is divided by the modelled generation to give the field efficiency the model implies.
2. The two other sites with a recorded flare flow (La Pradera and Presidente).
3. National disposal of the registry against the national series of the regulator.
4. Electricity the recovered gas would support at the seven largest sites, against the range of an earlier assessment of the major Colombian cities.
5. Route A cost for a single plant treating 700 m3/h of landfill gas, the scale of an operating European case study.

Writes results/validation.json. Run from the project root after the model:
    python scripts/13_validation.py
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MA = ROOT / "inputs" / "records"
RHO_N = 0.7168   # kg CH4 per normal m3 (0 C), as the operator's record
sys.path.insert(0, str(ROOT / "scripts"))


def q(a, p=(10, 50, 90)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, x)) for x in p]


def main():
    nat = pd.read_csv(RES / "mc_national.csv")
    site = pd.read_csv(RES / "site_results.csv", index_col=0)
    out = {}

    # 1. Doña Juana
    dj = pd.read_csv(MA / "dona_juana_biogas_year.csv")
    m = dj[dj.year.between(2021, 2025)]
    destroyed = float((m.flow_m3n_h * 8760.0 * 0.55 * RHO_N / 1000.0).mean())
    destroyed_alt = float((m.flow_m3n_h * 8760.0 * 0.50 * RHO_N / 1000.0).mean())
    g = nat["G_dj_kt"].to_numpy() * 1e3
    out["dj_gen_model_kt"] = [x / 1e3 for x in q(g)]
    out["dj_destroyed_kt"] = destroyed / 1e3
    out["dj_destroyed_alt_kt"] = destroyed_alt / 1e3
    out["dj_field_eff"] = q(destroyed / g)
    out["dj_field_eff_alt"] = q(destroyed_alt / g)

    # 2. recorded flares
    for sid, key in (("05237|LA PRADERA", "pradera"), ("76670|PRESIDENTE", "presidente"),
                     ("25099|NUEVO MONDONEDO", "mondonedo"), ("11001|DONA JUANA", "dj")):
        out[f"eta0_{key}"] = [float(site.loc[sid, f"eta0_{p}"]) for p in ("p10", "p50", "p90")]

    # 3. national totals
    reg = pd.read_csv(RES / "site_rows_assigned.csv").groupby("year")["tons"].sum()
    sspd = pd.read_csv(MA / "sspd_disposal_national_2015_2024.csv").set_index("year")["disposal_t"]
    d = (reg.reindex(sspd.index) / sspd - 1.0) * 100.0
    out["registry_vs_sspd_pct"] = {int(k): round(float(v), 2) for k, v in d.items()}
    out["registry_vs_sspd_mean_abs_pct"] = float(d.abs().mean())
    out["gen2024_model_kt"] = q(nat["gen2024_all_kt"])

    # 4. electricity at the seven largest sites, MW net
    top7 = site[site["status_2024"] == "active"].nlargest(7, "H2A_p50")
    mw = (top7["Pel_p50"] / 1e3).round(1)
    out["top7_MW"] = [float(x) for x in mw]
    out["top7_names"] = list(top7["display_name"])

    # 5. single European-scale plant, collection excluded
    M = importlib.import_module("11_lfg_h2_model")
    Qlfg, n = 700.0, 2000
    lc, E = [], []
    for _ in range(n):
        p = M.draw_params()
        R = np.full((1, M.NY), Qlfg * p["F"] * M.H)
        Rd = R.mean(axis=1)
        r = M.route_A(R, Rd, np.minimum(R, Rd[:, None]), p, "scale", False, 0.0, 0.0)
        lc.append(float(r["lcoh"][0]))
        E.append(float(r["E"][0]))
    out["plant700_lcohA"] = q(lc, (10, 50, 90))
    out["plant700_E"] = q(E, (10, 50, 90))
    out["plant700_H2_kg_d"] = float(Qlfg * 0.5 * M.H * np.median(E) * M.KG_H2_PER_M3 / 365.0)
    (RES / "validation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
