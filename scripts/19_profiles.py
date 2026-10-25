"""Annual operation of Route A at representative sites, at the default parameter values.

For Doña Juana, La Pradera and Colomba-El Guabal (the three largest sites, whose licenses end in 2026, 2027 and 2039), the script reports by year the methane recovered, the methane used and flared, the hydrogen produced and the utilization of the reformer train, in the base case and in the licensed-only variant, with the constant collection efficiency of the base case and with the collection efficiency by cover stage, without and with a minimum load of 50 % of the design flow.

Writes results/profiles.csv. Run from the project root:
    python scripts/19_profiles.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = [sys.argv[0], "1"]
M = importlib.import_module("11_lfg_h2_model")
DEF = importlib.import_module("15_defaults")
V = importlib.import_module("20_variants")

SITES = {"11001|DONA JUANA": "Doña Juana", "05237|LA PRADERA": "La Pradera",
         "76890|COLOMBA EL GUABAL": "Colomba-El Guabal"}


def main():
    p = DEF.central()
    G, Gb = V.generation(p)
    mcf, lo, md, hi, san = V.classes(p)
    eta_c = M.tri_ppf(p["u_eta"], lo, md, hi)
    eta0 = V.eta0_of(p, G, eta_c)
    Gh, Gbh = G[:, M.IH:], Gb[:, M.IH:]
    Hb_, groups, stg = V.dyn_matrices(p["u_eta"])
    _, _, Rdy, Rly = V.generation(p, dyn=(Hb_, groups))
    Rdy = np.where(san[:, None], Rdy, eta_c[:, None] * G)[:, M.IH:]
    Rly = np.where(san[:, None], Rly, eta_c[:, None] * (G - Gb))[:, M.IH:]
    cases = {"base": eta_c[:, None] * Gh, "lic": eta_c[:, None] * (Gh - Gbh),
             "dyn": Rdy, "dyn_lic": Rly}
    rows = []
    for case, R in cases.items():
        Rd = R.mean(axis=1)
        U = np.minimum(R, Rd[:, None])
        Ccf, Ocf = V.cf_cost(p, Rd, eta_c, eta0)
        for fmin in (0.0, V.FMIN_A):
            A = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf, {"fmin": fmin})
            for s, name in SITES.items():
                j = M.sidx[s]
                for t, y in enumerate(M.YEARS):
                    used = U[j, t] * A["on"][j, t]
                    rows.append(dict(site=name, case=case, fmin=fmin, year=int(y),
                                     gen_Mm3=Gh[j, t] / 1e6, rec_Mm3=R[j, t] / 1e6,
                                     used_Mm3=used / 1e6, flared_Mm3=(R[j, t] - used) / 1e6,
                                     h2_kt=A["H2_y"][j, t] / 1e6, util=used / Rd[j],
                                     design_Mm3=Rd[j] / 1e6, lcoh=float(A["lcoh"][j]),
                                     closure=int(V.closure[j])))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "profiles.csv", index=False, float_format="%.5g")
    print("cover-stage efficiencies at the mode:", np.round(stg, 3))
    for (site, case), g in out[out.fmin == 0].groupby(["site", "case"], sort=False):
        print(site, case, "closure", g.closure.iloc[0], "lcoh", round(g.lcoh.iloc[0], 2))
        print(g[["year", "rec_Mm3", "used_Mm3", "flared_Mm3", "h2_kt", "util"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
