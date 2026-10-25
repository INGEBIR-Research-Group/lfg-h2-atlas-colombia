"""Cost-capacity regression for reformer islands without carbon capture.

Fits ln C = b0 + n ln S + b1 EQUIP + b2 INST by ordinary least squares, where C is the cost in USD of 2023 and S the design output in kg of hydrogen per day, with total plant cost as the reference scope. The base sample is every published point without capture except the six Di Cicco points, whose island share of the plant cost had to be assumed.

Reports the fit with t-based confidence intervals, standard errors clustered by source, a leave-one-source-out check of the exponent and the specification variants quoted in the text. Writes

    results/capex_normalised.csv   every point on the common basis (Appendix A)
    results/capex_final.json       coefficients used by the model and the figure
    results/numbers_capex.csv      the values of the fit quoted in Table 13 and in the text

Run from the project root after scripts/07_capex_data.py:
    python scripts/08_capex_regression.py
"""

import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_data = importlib.import_module("07_capex_data")
CEPCI, FX = _data.CEPCI, _data.FX

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
S_REF = 1500.0
Z90 = 1.6449


def normalise():
    d = pd.read_csv(RESULTS / "capex_raw.csv")

    def usd2023(r):
        fx = FX.get((r.cur, int(r.year))) or FX.get((r.cur, 0))
        if fx is None:
            raise ValueError(f"no exchange rate for {r.cur} {r.year}")
        return r.cost * fx * CEPCI[2023] / CEPCI[int(r.year)]

    d["C"] = d.apply(usd2023, axis=1)
    d["S"] = d["kg_per_day"]
    d["lnC"], d["lnS"] = np.log(d["C"]), np.log(d["S"])
    d["EQUIP"] = (d["scope"] == "EQUIP").astype(int)
    d["INST"] = (d["scope"] == "INST").astype(int)
    d["BIO"] = (d["feed"] == "biogas").astype(int)
    d["assumed_split"] = d["id"].str.startswith("dicicco")
    d["in_base_fit"] = (d["ccs"] == 0) & ~d["assumed_split"]
    return d


def ols(sub, cols=("lnS", "EQUIP", "INST"), cluster=False):
    # an indicator that no point of the subset carries is dropped rather than left all zero
    cols = [c for c in cols if c == "lnS" or sub[c].astype(float).abs().sum() > 0]
    X = sm.add_constant(sub[list(cols)].astype(float))
    if cluster:
        g = sub["source"].astype("category").cat.codes
        return sm.OLS(sub["lnC"], X).fit(cov_type="cluster", cov_kwds={"groups": g})
    return sm.OLS(sub["lnC"], X).fit()


def main():
    d = normalise()
    base = d[d["in_base_fit"]].copy()
    m = ols(base)
    mc = ols(base, cluster=True)
    df_res = int(m.df_resid)
    t975 = stats.t.ppf(0.975, df_res)
    sigma = float(np.sqrt(m.scale))
    n = float(m.params["lnS"])

    loso = {}
    for src in sorted(base["source"].unique()):
        loso[src] = float(ols(base[base["source"] != src]).params["lnS"])

    var = {
        "with_assumed_splits": ols(d[(d["ccs"] == 0)]).params["lnS"],
        "tpc_only": ols(base[base["scope"] == "TPC"], cols=("lnS",)).params["lnS"],
        "natural_gas_only": ols(base[base["BIO"] == 0]).params["lnS"],
        "biogas_only": ols(base[base["BIO"] == 1]).params["lnS"],
    }
    # the equipment-scope study left out, and the installed scope pooled with total plant cost
    no_equip = base[base["EQUIP"] == 0]
    var["no_equipment_study"] = ols(no_equip).params["lnS"]
    var["installed_pooled"] = ols(base, cols=("lnS", "EQUIP")).params["lnS"]
    tpc_fit = ols(base[base["scope"] == "TPC"], cols=("lnS",))
    tpc_ci = [float(v) for v in tpc_fit.conf_int().loc["lnS"]]
    tpc_n = int((base["scope"] == "TPC").sum())
    tpc_src = int(base.loc[base["scope"] == "TPC", "source"].nunique())
    # scope sensitivity: the NETL bare erected cost read as an equipment cost instead
    alt = base.copy()
    bec = alt["id"] == "netl22_c1bec"
    alt.loc[bec, "EQUIP"], alt.loc[bec, "INST"] = 1, 0
    var["bec_as_equipment"] = ols(alt).params["lnS"]
    mbio = ols(base[base["BIO"] == 1])
    bio_only_ci = [float(v) for v in mbio.conf_int().loc["lnS"]]
    bio_only_n = int((base["BIO"] == 1).sum())
    n_equip = int(base["EQUIP"].sum())
    n_equip_src = int(base.loc[base["EQUIP"] == 1, "source"].nunique())
    base["ATR"] = (base["tech"] == "ATR").astype(int)
    ma = ols(base, cols=("lnS", "EQUIP", "INST", "ATR"))
    atr_premium = float(np.exp(ma.params["ATR"]) - 1.0)
    atr_p = float(ma.pvalues["ATR"])
    mb = ols(base, cols=("lnS", "EQUIP", "INST", "BIO"))
    bio_premium = float(np.exp(mb.params["BIO"]) - 1.0)
    bio_ci = [float(np.exp(v) - 1.0) for v in mb.conf_int().loc["BIO"]]

    lnC0 = float(m.params["const"] + n * np.log(S_REF))
    X0 = np.array([1.0, np.log(S_REF), 0.0, 0.0])
    se_fit = float(np.sqrt(X0 @ m.cov_params().to_numpy() @ X0))
    table = []
    for k, lab in [("const", "Constant"), ("lnS", "Scaling exponent"),
                   ("EQUIP", "Equipment scope"), ("INST", "Installed scope")]:
        lo, hi = m.conf_int().loc[k]
        table.append({"term": lab, "coef": float(m.params[k]), "se": float(m.bse[k]),
                      "se_cluster": float(mc.bse[k]), "ci_lo": float(lo), "ci_hi": float(hi),
                      "p": float(m.pvalues[k])})

    out = {
        "N": int(len(base)), "n_sources": int(base["source"].nunique()),
        "sources": sorted(base["source"].unique()),
        "const": float(m.params["const"]), "n": n, "n_se": float(m.bse["lnS"]),
        "n_ci_t": [n - t975 * float(m.bse["lnS"]), n + t975 * float(m.bse["lnS"])],
        "n_se_cluster": float(mc.bse["lnS"]),
        "df_resid": df_res, "t975": float(t975),
        "R2": float(m.rsquared), "R2_adj": float(m.rsquared_adj),
        "sigma": sigma, "band": float(np.exp(Z90 * sigma)),
        "equip": float(np.exp(m.params["EQUIP"])), "inst": float(np.exp(m.params["INST"])),
        "tpc_over_equip": float(np.exp(-m.params["EQUIP"])),
        "C1500": float(np.exp(lnC0)), "lnC1500_se_fit": se_fit,
        "S_min": float(base["S"].min()), "S_max": float(base["S"].max()),
        "orders_of_magnitude": float(np.log10(base["S"].max() / base["S"].min())),
        "loso": loso, "loso_min": min(loso.values()), "loso_max": max(loso.values()),
        "variants": {k: float(v) for k, v in var.items()},
        "bio_premium": bio_premium, "bio_premium_ci": bio_ci, "bio_p": float(mb.pvalues["BIO"]),
        "biogas_only_ci": bio_only_ci, "biogas_only_n": bio_only_n,
        "equip_points": n_equip, "equip_sources": n_equip_src,
        "atr_points": int(base["ATR"].sum()), "atr_premium": atr_premium, "atr_p": atr_p,
        "table": table,
    }
    d.to_csv(RESULTS / "capex_normalised.csv", index=False)
    (RESULTS / "capex_final.json").write_text(json.dumps(out, indent=1))

    def mac(name, val):
        return f"{name},{val}\n"

    tex = "name,value\n"
    tex += mac("CapexN", out["N"]) + mac("CapexSources", out["n_sources"])
    tex += mac("CapexR", f"{100 * out['R2']:.1f}") + mac("CapexRtwo", f"{out['R2']:.3f}")
    tex += mac("CapexSigma", f"{sigma:.3f}") + mac("CapexBand", f"{out['band']:.1f}")
    tex += mac("CapexExp", f"{n:.3f}") + mac("CapexExpSE", f"{out['n_se']:.3f}")
    tex += mac("CapexExpSEcl", f"{out['n_se_cluster']:.3f}")
    tex += mac("CapexExpLo", f"{out['n_ci_t'][0]:.3f}") + mac("CapexExpHi", f"{out['n_ci_t'][1]:.3f}")
    tex += mac("CapexLosoLo", f"{out['loso_min']:.3f}") + mac("CapexLosoHi", f"{out['loso_max']:.3f}")
    tex += mac("CapexVarSplits", f"{var['with_assumed_splits']:.3f}")
    tex += mac("CapexVarTPC", f"{var['tpc_only']:.3f}")
    tex += mac("CapexVarNG", f"{var['natural_gas_only']:.3f}")
    tex += mac("CapexVarBio", f"{var['biogas_only']:.2f}")
    tex += mac("CapexVarBEC", f"{var['bec_as_equipment']:.3f}")
    tex += mac("CapexVarNoEquip", f"{var['no_equipment_study']:.3f}")
    tex += mac("CapexVarInstPooled", f"{var['installed_pooled']:.3f}")
    tex += mac("CapexVarTPCLo", f"{tpc_ci[0]:.2f}") + mac("CapexVarTPCHi", f"{tpc_ci[1]:.2f}")
    tex += mac("CapexVarTPCN", tpc_n) + mac("CapexVarTPCSrc", tpc_src)
    tex += mac("CapexVarBioLo", f"{bio_only_ci[0]:.2f}") + mac("CapexVarBioHi", f"{bio_only_ci[1]:.2f}")
    tex += mac("CapexVarBioN", bio_only_n)
    tex += mac("CapexEquipN", n_equip) + mac("CapexEquipSrc", n_equip_src)
    tex += mac("CapexBioPrem", f"{100 * bio_premium:.0f}")
    tex += mac("CapexBioLo", f"{100 * bio_ci[0]:.0f}") + mac("CapexBioHi", f"{100 * bio_ci[1]:.0f}")
    tex += mac("CapexCzero", f"{out['C1500'] / 1e6:.2f}")
    tex += mac("CapexAtrN", out["atr_points"]) + mac("CapexAtrPrem", f"{100 * atr_premium:.0f}")
    tex += mac("CapexAtrP", f"{atr_p:.2f}") + mac("CapexBioP", f"{out['bio_p']:.2f}")
    tex += mac("CapexTPCoverEquip", f"{out['tpc_over_equip']:.1f}")
    tex += mac("CapexOrders", f"{out['orders_of_magnitude']:.1f}")
    tex += mac("CapexDF", df_res)
    for row, key in zip(table, ("Const", "Exp", "Equip", "Inst")):
        tex += mac(f"CapexT{key}", f"{row['coef']:.3f}") + mac(f"CapexT{key}SE", f"{row['se']:.3f}")
        tex += mac(f"CapexT{key}SEcl", f"{row['se_cluster']:.3f}")
        tex += mac(f"CapexT{key}Lo", f"{row['ci_lo']:.3f}") + mac(f"CapexT{key}Hi", f"{row['ci_hi']:.3f}")
        p = row["p"]
        tex += mac(f"CapexT{key}P", "<0.001" if p < 0.001 else f"{p:.3f}")
    (RESULTS / "numbers_capex.csv").write_text(tex)

    print(m.summary().tables[1])
    print(json.dumps({k: v for k, v in out.items() if k not in ("table", "sources")}, indent=1))


if __name__ == "__main__":
    main()
