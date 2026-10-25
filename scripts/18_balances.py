"""Mass and energy balances of Route A at two representative sites.

The balance of each site is taken from the equations of the model at the default parameter values (Table 12), at the design flow, and the island is set beside the thermodynamic balance of scripts/10_reformer_balance.py for the anchor closest in scale. Quantities are per kg of hydrogen delivered unless stated otherwise.

Writes results/balances.csv. Run from the project root after the model and scripts/10_reformer_balance.py:
    python scripts/18_balances.py [SITE_SMALL]
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
SMALL = next((a for a in sys.argv[1:] if "|" in a), None)
sys.argv = [sys.argv[0], "1"]
M = importlib.import_module("11_lfg_h2_model")
DEF = importlib.import_module("15_defaults")
V = importlib.import_module("20_variants")

RHO_CO2 = 1.872          # kg/m3 at 15 C and 101.325 kPa, as in scripts/15_defaults.py
LARGE = "11001|DONA JUANA"


def site_balance(j, p, R, Rd, U, Ccf, Ocf):
    A = M.route_A(R[j:j + 1], Rd[j:j + 1], U[j:j + 1], p, "scale", False, Ccf[j:j + 1], Ocf[j:j + 1])
    E = float(A["E"][0])
    rpsa = float(A["rpsa"][0])
    phi = float(A["phi"][0])
    ff = 1.0 / (1.0 + phi)
    a = float(A["a"][0])
    a_up = p["s_up"] / (1.0 - p["s_up"]) * ff
    a_eng = a - a_up
    xd = float(Rd[j]) / (1.0 + a)                  # island methane at the design flow, m3/yr
    h2 = float(A["H2_d"][0])                       # kg/yr at the design flow
    feed, fuel = xd * ff, xd * (1.0 - ff)
    lfg_feed = feed / (1.0 - p["s_up"]) / p["F"]   # landfill gas through the upgrading train
    conv = feed * p["eta_ref"]                     # feed methane converted
    co2_proc = M.RHO_CH4 * M.R_CO2 * conv          # kg/yr, before the adsorption unit
    co2_native = lfg_feed * (1.0 - p["F"]) * RHO_CO2
    # carbon burned: furnace fuel, unconverted feed methane in the tail gas and the generator, plus the CO2 that the raw landfill gas burned as fuel carries with it
    burned = fuel + feed * (1.0 - p["eta_ref"]) + a_eng * xd
    raw_burned = (fuel + a_eng * xd) / p["F"]
    co2_flue = burned * RHO_CO2 + raw_burned * (1.0 - p["F"]) * RHO_CO2
    # carbon closure: methane and CO2 of the recovered gas against what leaves, m3 of carbon
    c_in = float(Rd[j]) / p["F"]
    c_out = (co2_proc + co2_native + co2_flue) / RHO_CO2 + a_up * xd
    e_aux = p["e_aux"] * ff / p["F"] * xd          # kWh/yr
    return {
        "site": M.cat.loc[M.SITES[j], "display_name"],
        "design_H2_t_d": float(A["S"][0]) / 1e3,
        "recovered_CH4_m3_h": float(Rd[j]) / M.H,
        "LFG_to_upgrading_m3_h": lfg_feed / (M.H * M.AV),
        "E": E, "R_psa": rpsa, "eta_ref": p["eta_ref"], "phi_fuel": phi,
        "H2_formed_mol_per_mol_feed": 4.0 * p["eta_ref"],
        "H2_product_mol_per_mol_feed": 4.0 * p["eta_ref"] * rpsa,
        "recovered_CH4_m3_per_kg": float(Rd[j]) / h2,
        "lost_upgrading_m3_per_kg": a_up * xd / h2,
        "aux_generator_CH4_m3_per_kg": a_eng * xd / h2,
        "island_CH4_m3_per_kg": xd / h2,
        "feed_CH4_m3_per_kg": feed / h2,
        "fuel_CH4_m3_per_kg": fuel / h2,
        "aux_electricity_kWh_per_kg": e_aux / h2,
        "native_CO2_kg_per_kg": co2_native / h2,
        "process_CO2_kg_per_kg": co2_proc / h2,
        "flue_CO2_kg_per_kg": co2_flue / h2,
        "carbon_closure": c_out / c_in,
        "H2_yield_per_recovered_kg_per_m3": h2 / float(Rd[j]),
        "aux_generator": M.GEN_NAMES[int(A["tech"][0])],
        "LCOH": float(A["lcoh"][0]),
    }


def main():
    p = DEF.central()
    G, Gb = V.generation(p)
    mcf, lo, md, hi, san = V.classes(p)
    eta_c = M.tri_ppf(p["u_eta"], lo, md, hi)
    eta0 = V.eta0_of(p, G, eta_c)
    R = eta_c[:, None] * G[:, M.IH:]
    Rd = R.mean(axis=1)
    U = np.minimum(R, Rd[:, None])
    Ccf, Ocf = V.cf_cost(p, Rd, eta_c, eta0)
    A = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf)
    S = A["S"]
    small = SMALL
    if small is None:
        # the active sanitary landfill whose design output is closest to the small anchor
        cand = np.flatnonzero(M.active & M.sanitary & (S > 0))
        small = M.SITES[int(cand[np.argmin(np.abs(np.log(S[cand] / 1500.0)))])]
    rows = [site_balance(M.sidx[LARGE], p, R, Rd, U, Ccf, Ocf),
            site_balance(M.sidx[small], p, R, Rd, U, Ccf, Ocf)]
    sim = json.loads((ROOT / "results" / "reformer_balance.json").read_text())
    out = pd.DataFrame(rows).T
    out.columns = ["large", "small"]
    out.to_csv(ROOT / "results" / "balances.csv")
    print(out.to_string())
    print("small site:", small, M.cat.loc[small, ["display_name", "municipality", "tpd_2024"]].to_dict())
    print(json.dumps({k: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in s.items()
                          if not isinstance(v, list)} for k, s in sim.items() if k in ("small", "large")},
                     indent=1))


if __name__ == "__main__":
    main()
