"""Global sensitivity of the national results to the sampled parameters.

Partial rank correlation coefficients (PRCC) and standardized rank regression coefficients (SRRC) of each national output against the global parameters drawn in scripts/11_lfg_h2_model.py, computed on the full Monte Carlo ensemble. Site-level draws (composition perturbation, capture evidence) are not global parameters and are left in the residual.

Writes results/sensitivity.csv, whose leading parameters scripts/17_aggregate.py adds to results/numbers_results.csv.

Run from the project root after the model:
    python scripts/16_sensitivity.py
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

LABELS = {
    "u_eta": "Collection efficiency", "F": "Methane fraction of the gas",
    "u_E": "Overall reformer efficiency", "eta_ref": "Reforming conversion efficiency",
    "s_up": "Upgrading methane loss", "s_eng": "Engine methane slip",
    "s_mtur": "Microturbine methane slip", "sec": "Electrolyzer specific consumption",
    "nR": "Reformer scaling exponent", "z_cREF": "Level of the reformer capital",
    "cEL": "Electrolyzer capital", "nE": "Electrolyzer scaling exponent",
    "fL": "LFGcost-Web cost coefficients", "i": "Discount rate", "ccc": "Capture unit cost",
    "tns": "Carbon dioxide transport and storage", "q_regen": "Solvent regeneration heat",
    "e_cc": "Capture electricity", "pe": "Electricity price", "e_aux": "Auxiliary electricity",
    "n_up": "Upgrading scaling exponent below range", "s_burn": "Burner methane slip",
    "u_k1": "Decay constant of very fast waste", "u_k2": "Decay constant of medium fast waste",
    "u_k3": "Decay constant of medium slow waste", "u_k4": "Decay constant of slow waste",
    "l0m1": "Methane capacity of very fast waste", "l0m2": "Methane capacity of medium fast waste",
    "l0m3": "Methane capacity of medium slow waste", "l0m4": "Methane capacity of slow waste",
}
OUTPUTS = ["H2A_kt", "H2B_kt", "A_below_3", "lcohA_wmed", "lcohB_wmed", "lcohA_cc_wmed",
           "ciA_w", "ciB_w", "ciA_cc_w", "avoid_Mt", "rec_kt"]


def ranks(a):
    return pd.DataFrame(a).rank(method="average").to_numpy()


def prcc_srrc(X, y):
    Xr, yr = ranks(X), ranks(y.reshape(-1, 1))[:, 0]
    Xs = (Xr - Xr.mean(0)) / Xr.std(0)
    ys = (yr - yr.mean()) / yr.std()
    A = np.column_stack([np.ones(len(ys)), Xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    fit = A @ beta
    r2 = 1.0 - ((ys - fit) ** 2).sum() / (ys ** 2).sum()
    pr = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        o = np.delete(A, j + 1, axis=1)
        rx = Xs[:, j] - o @ np.linalg.lstsq(o, Xs[:, j], rcond=None)[0]
        ry = ys - o @ np.linalg.lstsq(o, ys, rcond=None)[0]
        pr[j] = np.corrcoef(rx, ry)[0, 1]
    return pr, beta[1:], r2


def main():
    p = pd.read_csv(RES / "mc_params.csv")
    n = pd.read_csv(RES / "mc_national.csv")
    keep = [c for c in p.columns if p[c].std() > 0]
    X = p[keep].to_numpy()
    rows = []
    for out in OUTPUTS:
        y = n[out].to_numpy()
        ok = np.isfinite(y)
        pr, sr, r2 = prcc_srrc(X[ok], y[ok])
        for c, a, b in zip(keep, pr, sr):
            rows.append({"output": out, "parameter": c, "label": LABELS.get(c, c),
                         "prcc": a, "srrc": b, "r2_rank": r2})
    s = pd.DataFrame(rows)
    s["abs_prcc"] = s["prcc"].abs()
    s = s.sort_values(["output", "abs_prcc"], ascending=[True, False])
    s.to_csv(RES / "sensitivity.csv", index=False, float_format="%.4f")
    for out in OUTPUTS:
        t = s[s.output == out].head(5)
        print(f"\n{out}  rank R2 = {t.r2_rank.iloc[0]:.2f}")
        print(t[["label", "prcc", "srrc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
