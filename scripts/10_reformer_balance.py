"""Mass and energy balance of a steam-reforming island fed with upgraded landfill gas.

The atlas describes the reformer island with three lumped parameters, the overall efficiency E (LHV of the hydrogen product over the LHV of all the methane consumed as feed and as furnace fuel), the reforming conversion eta_ref (hydrogen formed in the reformer and shift reactors over the stoichiometric 4 mol per mol of feed methane) and the hydrogen recovery R of the adsorption unit. This script derives E and eta_ref from a thermodynamic balance of the island for a small and a large plant, for an assumed R, so that the values assumed in the atlas can be checked against first principles.

Flowsheet: desulfurized biomethane is mixed with steam (steam-to-carbon ratio S/C), preheated and reformed in fired tubes to equilibrium at the outlet temperature (with an approach temperature); the syngas is cooled in a process-gas boiler, shifted in an adiabatic high-temperature shift reactor (and, in the large plant, a low-temperature one), cooled, dried and purified by pressure swing adsorption (PSA) with a hydrogen recovery R. The PSA tail gas and a supplementary fuel (raw landfill gas, as in the atlas) are burned in the reformer furnace with excess air. The radiant section supplies the reforming duty; the flue gas leaves it at the bridgewall temperature and is cooled in the convection section down to the stack temperature, preheating the mixed feed, raising and superheating the process steam and, in the large plant, preheating the combustion air. Heat that remains after these duties is reported as surplus; it is not credited to the hydrogen, because a landfill has no steam consumer.

Thermodynamic properties come from Cantera (GRI-Mech 3.0 species thermochemistry, ideal gas) and the IAPWS-IF97 water model of Cantera for steam.

Writes results/reformer_balance.json and reformer_balance_cases.csv. Run from the project root:
    python scripts/10_reformer_balance.py
"""
import json
import pathlib

import cantera as ct
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

SP = ["CH4", "H2O", "CO", "CO2", "H2", "N2", "O2", "AR"]
_gri = ct.Solution("gri30.yaml")
GAS = ct.Solution(thermo="ideal-gas", species=[_gri.species(n) for n in SP])
I = {s: k for k, s in enumerate(SP)}
T0 = 298.15
P_ATM = 101325.0
MW_H2 = 2.01588e-3


def vec(**kw):
    v = np.zeros(len(SP))
    for k, x in kw.items():
        v[I[k]] = x
    return v


def H(n, T):
    """Total enthalpy (J) of the ideal-gas mixture n (mol) at T (K), formation basis."""
    n = np.clip(n, 0.0, None)
    N = n.sum()
    if N <= 0:
        return 0.0
    GAS.TPX = T, P_ATM, n / N
    return GAS.enthalpy_mole / 1000.0 * N   # J/kmol -> J/mol


def T_of(n, Htarget, lo=250.0, hi=2600.0):
    return brentq(lambda T: H(n, T) - Htarget, lo, hi, xtol=1e-6)


def lhv_mol(species):
    """Lower heating value at 25 C, J/mol."""
    if species == "CH4":
        r, p = vec(CH4=1, O2=2), vec(CO2=1, H2O=2)
    elif species == "H2":
        r, p = vec(H2=1, O2=0.5), vec(H2O=1)
    elif species == "CO":
        r, p = vec(CO=1, O2=0.5), vec(CO2=1)
    else:
        raise ValueError(species)
    return H(r, T0) - H(p, T0)


LHV = {s: lhv_mol(s) for s in ("CH4", "H2", "CO")}


def elements(n):
    """Atoms of C, H, O, N, Ar in the mixture."""
    C = n[I["CH4"]] + n[I["CO"]] + n[I["CO2"]]
    Hh = 4 * n[I["CH4"]] + 2 * n[I["H2O"]] + 2 * n[I["H2"]]
    O = n[I["H2O"]] + n[I["CO"]] + 2 * n[I["CO2"]] + 2 * n[I["O2"]]
    return np.array([C, Hh, O, n[I["N2"]], n[I["AR"]]])


def reformer_equilibrium(n_in, T, P, approach):
    """Composition at chemical equilibrium of CH4, H2O, CO, CO2 and H2 at T - approach and P, with nitrogen and argon inert; returned in mol, conserving the elements of n_in."""
    GAS.TPX = T - approach, P, np.clip(n_in, 1e-12, None) / n_in.sum()
    GAS.equilibrate("TP")
    x = GAS.X.copy()
    e_in = elements(n_in)
    # scale by carbon conservation
    C_x = x[I["CH4"]] + x[I["CO"]] + x[I["CO2"]]
    N = e_in[0] / C_x
    return x * N


def k_wgs(T):
    """Equilibrium constant of CO + H2O = CO2 + H2 at T (K)."""
    GAS.TP = T, P_ATM
    g = GAS.standard_gibbs_RT
    dG = g[I["CO2"]] + g[I["H2"]] - g[I["CO"]] - g[I["H2O"]]
    return float(np.exp(-dG))


def shift_adiabatic(n_in, T_in, approach):
    """Adiabatic water-gas shift to equilibrium at the outlet temperature plus an approach (methane frozen). Returns the outlet composition and temperature."""
    h_in = H(n_in, T_in)
    d = vec(CO=-1, H2O=-1, CO2=1, H2=1)

    def resid(xi):
        n = n_in + xi * d
        T = T_of(n, h_in, T_in - 50.0, T_in + 400.0)
        K = k_wgs(T + approach)
        return n[I["CO2"]] * n[I["H2"]] - K * n[I["CO"]] * n[I["H2O"]]

    xi_max = min(n_in[I["CO"]], n_in[I["H2O"]]) * (1 - 1e-9)
    xi = brentq(resid, 0.0, xi_max, xtol=1e-10)
    n = n_in + xi * d
    return n, T_of(n, h_in, T_in - 50.0, T_in + 400.0)


def psat_water(T):
    w = ct.Water()
    w.TQ = T, 0.0
    return w.P


def dry(n, T, P):
    """Gas after condensation at T and P: the water left in the vapor."""
    ps = psat_water(T)
    n = n.copy()
    other = n.sum() - n[I["H2O"]]
    v = ps / (P - ps) * other
    cond = max(n[I["H2O"]] - v, 0.0)
    n[I["H2O"]] = min(n[I["H2O"]], v)
    return n, cond


def steam_dh(P_steam, T_steam, T_bfw=298.15):
    """J per mol of water raised from boiler feed water at T_bfw to steam at P and T."""
    w = ct.Water()
    w.TP = T_bfw, P_steam
    h0 = w.enthalpy_mass
    w.TP = T_steam, P_steam
    return (w.enthalpy_mass - h0) * 18.015e-3


def combustion(fuel, air_ratio):
    """Products of complete combustion of the fuel mixture with air_ratio times the stoichiometric air (21 % O2, 78 % N2, 1 % Ar); returns (air, products)."""
    o2 = 2 * fuel[I["CH4"]] + 0.5 * fuel[I["H2"]] + 0.5 * fuel[I["CO"]]
    o2_air = air_ratio * o2
    air = vec(O2=o2_air, N2=o2_air * 78.0 / 21.0, AR=o2_air * 1.0 / 21.0)
    prod = vec(CO2=fuel[I["CO2"]] + fuel[I["CH4"]] + fuel[I["CO"]],
               H2O=fuel[I["H2O"]] + 2 * fuel[I["CH4"]] + fuel[I["H2"]],
               N2=fuel[I["N2"]] + air[I["N2"]], AR=air[I["AR"]],
               O2=o2_air - o2)
    return air, prod


def island(case):
    """Balance of the island per mol of feed methane for the parameters in case."""
    c = dict(case)
    feed = vec(CH4=c["feed_ch4"], CO2=c["feed_co2"], N2=c["feed_n2"])
    feed = feed / feed[I["CH4"]]                     # per mol of feed CH4
    steam = vec(H2O=c["sc"] * 1.0)                   # S/C on methane carbon
    mix = feed + steam
    # mixed feed preheated to the reformer inlet, reformed to equilibrium at the outlet
    T_in, T_out = c["T_mix"] + 273.15, c["T_ref"] + 273.15
    syn = reformer_equilibrium(mix, T_out, c["P_ref"], c["approach_ref"])
    Q_ref = H(syn, T_out) - H(mix, T_in)             # reforming duty, J
    ch4_conv = 1 - syn[I["CH4"]] / mix[I["CH4"]]
    # process-gas boiler and shift reactors
    T_hts = c["T_hts"] + 273.15
    Q_pgb = H(syn, T_out) - H(syn, T_hts)
    s1, T1 = shift_adiabatic(syn, T_hts, c["approach_shift"])
    Q_mid = 0.0
    sN, TN = s1, T1
    if c.get("T_lts"):
        T_lts = c["T_lts"] + 273.15
        Q_mid = H(s1, T1) - H(s1, T_lts)
        sN, TN = shift_adiabatic(s1, T_lts, c["approach_shift"])
    # useful process-side heat below the last shift: down to the boiler-feed-water limit
    T_use = c["T_proc_use"] + 273.15
    Q_low = max(H(sN, TN) - H(sN, T_use), 0.0)
    co2_process = sN[I["CO2"]]                       # concentrated CO2 before the PSA
    # condensation and PSA
    gas_dry, cond = dry(sN, 313.15, c["P_ref"] * 0.95)
    h2_in_psa = gas_dry[I["H2"]]
    h2_prod = c["R_psa"] * h2_in_psa
    tail = gas_dry.copy()
    tail[I["H2"]] -= h2_prod
    # furnace: tail gas plus a supplementary fuel (raw landfill gas)
    sup_unit = vec(CH4=c["sup_ch4"], CO2=c["sup_co2"], N2=c["sup_n2"])
    sup_unit = sup_unit / sup_unit[I["CH4"]]         # per mol of supplementary CH4
    T_air_max = c["T_air_max"] + 273.15
    T_bw, T_stack = c["T_bw"] + 273.15, c["T_stack"] + 273.15
    T_fuel = 298.15
    w_steam = steam_dh(c["P_steam"], c["T_steam"] + 273.15)
    Q_steam = c["sc"] * w_steam                       # raising the process steam
    # the feed methane is preheated for desulfurization before the steam is added, and the mixed stream is then heated to the reformer inlet in the convection section
    T_ds = c["T_desulf"] + 273.15
    Q_feedpre = H(feed, T_ds) - H(feed, 298.15)
    steam_T = c["T_steam"] + 273.15
    Q_mixpre = H(mix, T_in) - (H(feed, T_ds) + H(steam, steam_T))

    def fired(f_sup, T_air):
        fuel = tail + f_sup * sup_unit
        air, prod = combustion(fuel, c["air_ratio"])
        h_react = H(fuel, T_fuel) + H(air, T_air)
        lhv_fired = (fuel[I["CH4"]] * LHV["CH4"] + fuel[I["H2"]] * LHV["H2"] + fuel[I["CO"]] * LHV["CO"])
        loss = c["loss"] * lhv_fired
        Q_rad = h_react - H(prod, T_bw) - loss
        Q_conv = H(prod, T_bw) - H(prod, T_stack)
        Q_air = H(air, T_air) - H(air, 298.15)
        return Q_rad, Q_conv, Q_air, lhv_fired, prod, air

    def solve(T_air):
        f = brentq(lambda fs: fired(fs, T_air)[0] - Q_ref, -5.0, 50.0, xtol=1e-10)
        Q_rad, Q_conv, Q_air, lhv_fired, prod, air = fired(f, T_air)
        demand = Q_feedpre + Q_mixpre + Q_steam + Q_air
        avail = Q_conv + Q_pgb + Q_mid + Q_low
        return f, avail - demand, Q_conv, Q_air, demand, avail, lhv_fired

    T_air = 298.15
    f, surplus, Q_conv, Q_air, demand, avail, lhv_fired = solve(T_air)
    if T_air_max > 298.15 and surplus > 0 and f > 0:
        # preheat the combustion air until the convection and process heat just cover the duties, which minimizes firing when no steam can be exported; the preheat stops where the tail gas alone would supply the radiant duty
        T_hi = T_air_max
        if solve(T_hi)[0] < 0:
            T_hi = brentq(lambda Ta: solve(Ta)[0], 298.15, T_air_max, xtol=1e-3)
        g = lambda Ta: solve(Ta)[1]
        T_air = brentq(g, 298.15, T_hi, xtol=1e-3) if g(T_hi) < 0 else T_hi
        f, surplus, Q_conv, Q_air, demand, avail, lhv_fired = solve(T_air)
    extra = 0.0
    if surplus < 0:
        # a deficit is covered by burning more supplementary fuel in the furnace, whose flue gas recovers heat down to the stack temperature
        def deficit(fx):
            _, sp, *_ = solve_extra(fx)
            return sp

        def solve_extra(fx):
            fuel = tail + (f + fx) * sup_unit
            air, prod = combustion(fuel, c["air_ratio"])
            lhv_f = (fuel[I["CH4"]] * LHV["CH4"] + fuel[I["H2"]] * LHV["H2"] + fuel[I["CO"]] * LHV["CO"])
            h_react = H(fuel, T_fuel) + H(air, T_air)
            q_total = h_react - H(prod, T_stack) - c["loss"] * lhv_f - Q_ref
            q_air = H(air, T_air) - H(air, 298.15)
            dem = Q_feedpre + Q_mixpre + Q_steam + q_air
            av = q_total + Q_pgb + Q_mid + Q_low
            return lhv_f, av - dem, q_total, q_air, dem, av
        extra = brentq(deficit, 0.0, 20.0, xtol=1e-10)
        lhv_fired, surplus, Q_conv, Q_air, demand, avail = solve_extra(extra)
    sup_ch4 = f + extra
    ch4_total = 1.0 + sup_ch4
    E = h2_prod * LHV["H2"] / (ch4_total * LHV["CH4"])
    # an export credit, as some sources apply: surplus heat valued as boiler fuel at 90 %
    E_credit = h2_prod * LHV["H2"] / (ch4_total * LHV["CH4"] - max(surplus, 0.0) / 0.90)
    eta_ref = h2_in_psa / 4.0          # hydrogen formed, before the adsorption unit
    yield_prod = h2_prod / 4.0         # hydrogen delivered, after it
    # net process water: steam fed minus the condensate that can be returned to the boiler
    water_net_mol = c["sc"] - cond * c["condensate_return"]
    kg_h2 = h2_prod * MW_H2
    return {
        "case": c["name"], "scale_t_d": c["scale_t_d"],
        "T_ref_C": c["T_ref"], "P_ref_MPa": c["P_ref"] / 1e6, "S_C": c["sc"], "R_psa": c["R_psa"],
        "ch4_conversion": ch4_conv,
        "dry_syngas_H2_frac": gas_dry[I["H2"]] / gas_dry.sum(),
        "h2_before_psa_mol_per_mol": h2_in_psa,
        "h2_product_mol_per_mol": h2_prod,
        "eta_ref": eta_ref,
        "yield_product": yield_prod,
        "phi_fuel": sup_ch4,
        "phi_from_model_eq": 4.0 * MW_H2 * 0.678 * 119.96 / (16.043e-3 * 33.93) * c["R_psa"] * eta_ref / E - 1.0,
        "E": E, "E_with_heat_credit": E_credit,
        "T_air_C": T_air - 273.15,
        "reforming_duty_MJ_per_kg_H2": Q_ref / kg_h2 / 1e6,
        "surplus_heat_MJ_per_kg_H2": max(surplus, 0.0) / kg_h2 / 1e6,
        "tail_gas_LHV_share_of_firing": (tail[I["CH4"]] * LHV["CH4"] + tail[I["H2"]] * LHV["H2"]
                                         + tail[I["CO"]] * LHV["CO"]) / lhv_fired,
        "process_CO2_kg_per_kg_H2": co2_process * 44.01e-3 / kg_h2,
        "flue_CO2_kg_per_kg_H2": (tail[I["CO2"]] + tail[I["CH4"]] + tail[I["CO"]]
                                  + sup_ch4 * (1 + c["sup_co2"] / c["sup_ch4"])) * 44.01e-3 / kg_h2
                                 - co2_process * 44.01e-3 / kg_h2 * 0.0,
        "steam_kg_per_kg_H2": c["sc"] * 18.015e-3 / kg_h2,
        "water_net_kg_per_kg_H2": water_net_mol * 18.015e-3 / kg_h2,
        "feed_ch4_m3_per_kg_H2": 1.0 / (kg_h2 * 42.29),
        "total_ch4_m3_per_kg_H2": ch4_total / (kg_h2 * 42.29),
    }


# 1 m3 at 15 C and 101.325 kPa holds 42.29 mol of ideal gas
BASE = dict(feed_ch4=0.97, feed_co2=0.02, feed_n2=0.01,      # biomethane after upgrading
            sup_ch4=0.50, sup_co2=0.48, sup_n2=0.02,         # raw landfill gas as fuel
            approach_ref=10.0, approach_shift=10.0, condensate_return=0.9,
            T_desulf=370.0)
SMALL = dict(BASE, name="small", scale_t_d=1.5, sc=3.0, T_ref=820.0, P_ref=2.0e6, T_mix=550.0,
             T_hts=350.0, T_lts=None, T_proc_use=150.0, R_psa=0.78, air_ratio=1.20,
             T_air_max=25.0, T_bw=950.0, T_stack=250.0, loss=0.04, P_steam=2.5e6, T_steam=350.0)
LARGE = dict(BASE, name="large", scale_t_d=50.0, sc=3.0, T_ref=860.0, P_ref=2.5e6, T_mix=620.0,
             T_hts=350.0, T_lts=200.0, T_proc_use=130.0, R_psa=0.88, air_ratio=1.10,
             T_air_max=500.0, T_bw=1000.0, T_stack=140.0, loss=0.02, P_steam=4.0e6, T_steam=400.0)


def main():
    rows = []
    for base in (SMALL, LARGE):
        rows.append(island(base))
        # one-at-a-time ranges of the design choices that most affect E and eta_ref
        for key, vals in (("sc", (2.5, 3.5)), ("T_ref", (base["T_ref"] - 30, base["T_ref"] + 30)),
                          ("R_psa", (base["R_psa"] - 0.05, min(base["R_psa"] + 0.04, 0.92))),
                          ("T_stack", (base["T_stack"] - 40, base["T_stack"] + 60)),
                          ("sup_ch4", (0.97,)), ("T_air_max", (25.0, 550.0))):
            for v in vals:
                cc = dict(base)
                cc[key] = v
                if key == "sup_ch4":
                    cc.update(sup_ch4=0.97, sup_co2=0.02, sup_n2=0.01)
                cc["name"] = f"{base['name']}:{key}={v}"
                rows.append(island(cc))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reformer_balance_cases.csv", index=False, float_format="%.5g")
    summary = {}
    for nm in ("small", "large"):
        d = df[df["case"].str.startswith(nm)]
        b = df[df["case"] == nm].iloc[0]
        summary[nm] = {k: float(b[k]) for k in ("E", "E_with_heat_credit", "eta_ref", "yield_product",
                                                 "phi_fuel", "phi_from_model_eq", "R_psa",
                                                 "ch4_conversion", "h2_before_psa_mol_per_mol",
                                                 "process_CO2_kg_per_kg_H2", "water_net_kg_per_kg_H2",
                                                 "steam_kg_per_kg_H2", "surplus_heat_MJ_per_kg_H2",
                                                 "T_air_C", "reforming_duty_MJ_per_kg_H2",
                                                 "tail_gas_LHV_share_of_firing")}
        summary[nm]["E_range"] = [float(d["E"].min()), float(d["E"].max())]
        summary[nm]["eta_ref_range"] = [float(d["eta_ref"].min()), float(d["eta_ref"].max())]
        summary[nm]["yield_range"] = [float(d["yield_product"].min()), float(d["yield_product"].max())]
        summary[nm]["phi_range"] = [float(d["phi_fuel"].min()), float(d["phi_fuel"].max())]
        summary[nm]["co2_range"] = [float(d["process_CO2_kg_per_kg_H2"].min()),
                                    float(d["process_CO2_kg_per_kg_H2"].max())]
    summary["LHV_J_per_mol"] = LHV
    (OUT / "reformer_balance.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 220)
    print(df[["case", "E", "E_with_heat_credit", "eta_ref", "yield_product", "phi_fuel", "phi_from_model_eq", "ch4_conversion",
              "h2_before_psa_mol_per_mol", "T_air_C", "surplus_heat_MJ_per_kg_H2",
              "process_CO2_kg_per_kg_H2", "water_net_kg_per_kg_H2"]].round(3).to_string())


if __name__ == "__main__":
    main()
