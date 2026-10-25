"""Checks of the reformer cost-capacity regression.

1. Prediction intervals of the total plant cost at representative capacities, from the fit of Table 13 (t-based, 17 degrees of freedom), at 90 % and 95 %.
2. The fit without the two autothermal-reforming points.
3. A weighted fit in which every study counts equally (weights 1/k for a study with k points), since four studies contribute several variants of one design.
4. A mixed model with a random intercept by study, which separates the scatter between studies from the scatter within them.
5. The leave-one-study-out range of the level at 1,500 kg/d, beside that of the exponent.

Writes results/capex_checks.json. Run from the project root after scripts/08_capex_regression.py:
    python scripts/09_capex_checks.py
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
S_REF = 1500.0
CAPS = [100.0, 500.0, 1500.0, 5000.0, 15000.0, 50000.0]


def fit(sub, weights=None, cols=("lnS", "EQUIP", "INST")):
    cols = [c for c in cols if c == "lnS" or sub[c].astype(float).abs().sum() > 0]
    X = sm.add_constant(sub[cols].astype(float))
    if weights is None:
        return sm.OLS(sub["lnC"], X).fit(), cols
    return sm.WLS(sub["lnC"], X, weights=weights).fit(), cols


def level(m, cols, S=S_REF):
    x = np.array([1.0] + [np.log(S) if c == "lnS" else 0.0 for c in cols])
    return float(np.exp(x @ m.params.to_numpy()))


def main():
    d = pd.read_csv(RES / "capex_normalised.csv")
    base = d[d["in_base_fit"]].copy()
    out = {}
    m, cols = fit(base)
    out["base"] = dict(n=float(m.params["lnS"]), C1500=level(m, cols), sigma=float(np.sqrt(m.scale)),
                       N=int(len(base)))
    # 1. prediction intervals of a total-plant-cost island
    pi = []
    for S in CAPS:
        x = pd.DataFrame({"const": [1.0], "lnS": [np.log(S)], "EQUIP": [0.0], "INST": [0.0]})
        pr = m.get_prediction(x[["const"] + cols])
        row = {"S_kg_d": S, "C_MUSD": float(np.exp(pr.predicted_mean[0])) / 1e6}
        for lev in (0.90, 0.95):
            lo, hi = pr.conf_int(obs=True, alpha=1 - lev)[0]
            row[f"pi{int(lev * 100)}_lo"] = float(np.exp(lo)) / 1e6
            row[f"pi{int(lev * 100)}_hi"] = float(np.exp(hi)) / 1e6
            row[f"pi{int(lev * 100)}_factor"] = float(np.exp(0.5 * (hi - lo)))
            clo, chi = pr.conf_int(obs=False, alpha=1 - lev)[0]
            row[f"ci{int(lev * 100)}_factor"] = float(np.exp(0.5 * (chi - clo)))
        pi.append(row)
    out["prediction_intervals"] = pi
    # points inside the band of the Monte Carlo (5th to 95th percentile of the lognormal)
    resid = m.resid.to_numpy()
    out["share_resid_within_mc_band"] = float(np.mean(np.abs(resid) <= 1.6449 * np.sqrt(m.scale)))
    # 2. without the autothermal points
    noatr = base[base["tech"] != "ATR"]
    m2, c2 = fit(noatr)
    out["no_atr"] = dict(n=float(m2.params["lnS"]), C1500=level(m2, c2), sigma=float(np.sqrt(m2.scale)),
                         N=int(len(noatr)), n_ci=[float(v) for v in m2.conf_int().loc["lnS"]])
    # 3. every study with the same total weight
    k = base.groupby("source")["id"].transform("count").to_numpy(float)
    m3, c3 = fit(base, weights=1.0 / k)
    out["study_weighted"] = dict(n=float(m3.params["lnS"]), C1500=level(m3, c3),
                                 n_ci=[float(v) for v in m3.conf_int().loc["lnS"]],
                                 points_per_study=base.groupby("source")["id"].count().to_dict())
    # 4. random intercept by study
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = smf.mixedlm("lnC ~ lnS + EQUIP + INST", base, groups=base["source"]).fit(reml=True)
    tau2 = float(mm.cov_re.iloc[0, 0])
    s2 = float(mm.scale)
    out["mixed"] = dict(n=float(mm.params["lnS"]), n_se=float(mm.bse["lnS"]),
                        C1500=float(np.exp(mm.params["Intercept"] + mm.params["lnS"] * np.log(S_REF))),
                        tau=float(np.sqrt(tau2)), sigma_within=float(np.sqrt(s2)),
                        share_between=tau2 / (tau2 + s2), sigma_total=float(np.sqrt(tau2 + s2)))
    # 5. leave one study out, level at 1,500 kg/d
    lv = {}
    for src in sorted(base["source"].unique()):
        mi, ci = fit(base[base["source"] != src])
        lv[src] = level(mi, ci)
    out["loso_C1500_MUSD"] = {k_: v / 1e6 for k_, v in lv.items()}
    out["loso_C1500_range_MUSD"] = [min(lv.values()) / 1e6, max(lv.values()) / 1e6]
    # points between 1.5 and 70 t/d, a range that covers the design outputs of the largest Colombian sites (Doña Juana, 52.7 t/d at the default values)
    mid = base[(base["S"] >= 1500.0) & (base["S"] <= 70000.0)]
    out["points_1p5_to_70t"] = int(len(mid))
    out["points_1p5_to_70t_ids"] = list(mid["id"])
    out["t975"] = float(stats.t.ppf(0.975, m.df_resid))
    (RES / "capex_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
