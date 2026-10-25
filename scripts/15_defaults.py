"""Derived quantities at the default (modal) parameter values, quoted in the Methods.

Writes results/numbers_defaults.csv. Run from the project root after the model inputs exist:
    python scripts/15_defaults.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
M = importlib.import_module("11_lfg_h2_model")


def central():
    p = M.draw_params()
    tq = lambda lo, m, hi: (m - lo) / (hi - lo)  # noqa: E731
    p.update(u_k=np.full(4, 0.5), l0m=np.ones(4), u_eta=tq(0.41, 0.60, 0.71), F=0.50,
             u_E=tq(*M.E_SMALL), eta_ref=0.75, s_up=0.005, s_eng=0.016, s_mtur=0.003, sec=53.0,
             nR=M.capex["n"], z_cREF=0.0, cREF=M.capex["C1500"], cEL=1.8e6, nE=0.78, fL=1.0,
             i=0.09, ccc=60.0 * M.ESC[2018], tns=40.0, q_regen=2.9, e_cc=0.12, pe=0.088,
             e_aux=0.25, n_up=0.65, s_burn=M.S_BURN[1], u_psa=0.5, co2_ts=M.CO2_TS[1],
             z_comp=np.zeros((M.NS, 4)), u_cap=np.full(M.NS, 0.5), u_unrec=np.full(M.NS, 0.5))
    return p


def national_generation(lag=None, full_year=False):
    """National methane generation of the active sites, 2026-2040 mean, at modal decay parameters and without composition perturbation, in m3/yr. full_year=True counts a full year of decay in the year of deposit."""
    G = np.zeros((M.NS, len(M.EVAL)))
    for z in range(5):
        sel = M.zone == z
        if not sel.any():
            continue
        for c in range(4):
            k = M.K[c, z]
            if full_year:
                D = np.where(M.AGE >= 0, np.exp(-k * np.clip(M.AGE, 0, None)), 0.0)
            else:
                D = M.decay_kernel(k, lag)
            G[sel] += k * M.L0C[c] * (M.Wc[sel, c, :] @ D)
    G *= M.mcf0[:, None]
    return float(G[M.active][:, M.IH:].sum(axis=0).mean())


def main():
    p = central()
    out = {}

    def mac(k, v, f="{:.3f}"):
        out[k] = f.format(v)

    # reformer efficiency at the two anchors and the fuel share it implies
    for lab, E, rp in (("Small", M.E_SMALL[1], M.R_PSA_SMALL[1]), ("Large", M.E_LARGE[1], M.R_PSA_LARGE[1])):
        phi = 1.2051 * rp * p["eta_ref"] / E - 1.0
        mac(f"DPsa{lab}", rp, "{:.2f}")
        mac(f"DYieldFeed{lab}", 4.0 * rp * p["eta_ref"], "{:.2f}")
        mac(f"DE{lab}", E, "{:.2f}")
        mac(f"DPhi{lab}", phi, "{:.2f}")
        mac(f"DFuel{lab}", 100 * phi / (1 + phi), "{:.0f}")
        mac(f"DYieldIsl{lab}", E * M.KG_H2_PER_M3, "{:.3f}")
    mac("DQsmall", M.Q_SMALL, "{:.0f}")
    mac("DQlarge", M.Q_LARGE, "{:.0f}")
    phi_s = 1.2051 * M.R_PSA_SMALL[1] * p["eta_ref"] / M.E_SINGH - 1.0
    mac("DPhiSingh", phi_s, "{:.2f}")
    mac("DYieldSingh", M.E_SINGH * M.KG_H2_PER_M3, "{:.3f}")
    # yield per unit of recovered methane for three plant sizes, constant flow
    sizes = {"Small": 1500.0, "Mid": 15000.0, "Large": 60000.0}   # kg H2/d nameplate, roughly
    for lab, kgd in sizes.items():
        Rch4 = kgd / (0.65 * M.KG_H2_PER_M3) * 365.0 * 1.12      # m3 CH4/yr recovered
        R = np.full((1, M.NY), Rch4)
        Rd = R.mean(axis=1)
        A = M.route_A(R, Rd, np.minimum(R, Rd[:, None]), p, "scale", False, 0.0, 0.0)
        a_eng = A["a"][0] - p["s_up"] / (1 - p["s_up"]) / (1 + A["phi"][0])
        mac(f"DYieldRec{lab}", A["H2_y"][0].mean() / Rch4, "{:.3f}")
        mac(f"DAux{lab}", 100 * a_eng / (1 + A["a"][0]), "{:.1f}")
        mac(f"DEat{lab}", A["E"][0], "{:.2f}")
        mac(f"DCiA{lab}", A["ci"][0], "{:.2f}")
        mac(f"DKgd{lab}", A["S"][0], "{:.0f}")
        Acc = M.route_A(R, Rd, np.minimum(R, Rd[:, None]), p, "scale", True, 0.0, 0.0)
        mac(f"DCiAcc{lab}", Acc["ci"][0], "{:.2f}")
        mac(f"DCOtwo{lab}", Acc["co2"][0] / Acc["H2_y"][0].mean(), "{:.2f}")
        mac(f"DCcPenalty{lab}", 100 * (1 - Acc["H2_y"][0].mean() / A["H2_y"][0].mean()), "{:.0f}")
        if lab == "Mid":
            # how much of the reforming intensity is the upgrading loss, and what an off-gas oxidizer (0.1 %) or an adsorption unit without one (2 %) would give
            for tag, sup in (("NoUp", 0.0), ("Ox", 0.001), ("Psa", 0.020)):
                q = dict(p, s_up=sup)
                Aq = M.route_A(R, Rd, np.minimum(R, Rd[:, None]), q, "scale", False, 0.0, 0.0)
                mac(f"DCiA{tag}Mid", Aq["ci"][0], "{:.2f}")
            mac("DShareUpMid", 100 * (1 - float(out["DCiANoUpMid"]) / A["ci"][0]), "{:.0f}")
            # biogenic CO2 that the upgrading train separates from the raw feed gas, which the capture variant does not credit (density of CO2 at 15 C and 101.325 kPa)
            co2_up = A["Qfeed"][0] * M.H * M.AV * (1 - p["F"]) * 1.872 / A["H2_y"][0].mean()
            mac("DCOtwoUpMid", co2_up, "{:.1f}")
    # route B yields per generator
    for name in M.GEN_NAMES:
        HR, epar = M.GEN[name][0], M.GEN[name][1]
        y = M.HHV_CH4 / HR * epar / p["sec"]
        mac(f"DYieldB{name.capitalize()}", y, "{:.3f}")
        slip = {"ENG": p["s_eng"], "SENG": p["s_eng"], "TUR": p["s_tur"], "MTUR": p["s_mtur"]}[name]
        mac(f"DCiB{name.capitalize()}", M.RHO_CH4 * M.GWP * slip * HR * p["sec"] / (M.HHV_CH4 * epar), "{:.2f}")
    mac("DGridCi", 0.220 * p["sec"], "{:.1f}")
    # engine slip at which electrolysis on engine power just meets the Colombian ceiling
    for name in ("ENG", "SENG"):
        HR, epar = M.GEN[name][0], M.GEN[name][1]
        per_slip = M.RHO_CH4 * M.GWP * HR * p["sec"] / (M.HHV_CH4 * epar)
        mac(f"DSlipStar{name.capitalize()}", 100 * 3.6 / per_slip, "{:.1f}")
    # ---------------------------------------------------------------- further checks
    # decay timing: the six-month lag against no lag, a one-year lag and a full year of decay
    g05 = national_generation()
    mac("DLagZeroPct", 100 * (national_generation(0.0) / g05 - 1), "{:+.1f}")
    mac("DLagOnePct", 100 * (national_generation(1.0) / g05 - 1), "{:+.1f}")
    mac("DGenFullYearPct", 100 * (national_generation(full_year=True) / g05 - 1), "{:+.1f}")
    # burner slip of the reformer furnace and the capture boiler, mid-size site
    Rch4 = sizes["Mid"] / (0.65 * M.KG_H2_PER_M3) * 365.0 * 1.12
    R = np.full((1, M.NY), Rch4)
    Rd = R.mean(axis=1)
    U = np.minimum(R, Rd[:, None])
    ci = {}
    for tag, sb in (("0", 0.0), ("def", M.S_BURN[1]), ("hi", M.S_BURN[2])):
        for cc in (False, True):
            ci[(tag, cc)] = M.route_A(R, Rd, U, dict(p, s_burn=sb), "scale", cc, 0.0, 0.0)["ci"][0]
    mac("DCiBurnMid", ci[("def", False)] - ci[("0", False)], "{:.3f}")
    mac("DCiBurnHiMid", ci[("hi", False)] - ci[("0", False)], "{:.3f}")
    mac("DCiBurnCcHiMid", ci[("hi", True)] - ci[("0", True)], "{:.3f}")
    mac("DSburnPct", 100 * M.S_BURN[1], "{:.3f}")
    mac("DSburnHiPct", 100 * M.S_BURN[2], "{:.3f}")
    # site-level quantities at the modal values, for the upgrading and generator checks
    base = M.one_draw(p)
    act = M.active
    # the gas turbine as least-cost generator of Route B: scan of the recovered methane flow, with the collection cost of a site of that flow from the active sites (log-log fit)
    ok = act & (base["rec"] > 0)
    fit = np.polyfit(np.log(base["rec"][ok]), np.log(M.Nw[ok]), 1)
    flows = np.geomspace(300.0, 20000.0, 600)               # m3 CH4/h recovered
    Rs = flows * M.H
    Us = np.repeat(Rs[:, None], M.NY, axis=1)
    Nw_s = np.exp(np.polyval(fit, np.log(Rs)))
    Ccf_s = (2.00e4 + Nw_s * (279.0 * (M.DEPTH - 3.05) + 1.77e4)
             + 3329.0 * (Rs / p["F"] / M.H) ** 0.61) * M.ESC[2013]
    Ocf_s = (2600.0 * Nw_s + 5100.0) * M.ESC[2013]
    Bs = M.route_B(Rs, Us, p, Ccf_s, Ocf_s)
    tur = Bs["tech"] == M.GEN_NAMES.index("TUR")
    if tur.any():
        qt = float(flows[np.argmax(tur)])
        P_t = qt * M.HHV_CH4 / (M.GEN["TUR"][0] * M.GEN["TUR"][2])     # kW gross
        mac("DTurQ", qt, "{:.0f}")
        mac("DTurQlfg", qt / p["F"], "{:.0f}")
        mac("DTurMW", P_t / 1e3, "{:.1f}")
        mac("DTurMWnet", P_t * M.GEN["TUR"][1] / 1e3, "{:.1f}")
    eng = Bs["tech"] == M.GEN_NAMES.index("ENG")
    mac("DEngQlo", float(flows[np.argmax(eng)]) if eng.any() else float("nan"), "{:.0f}")
    # cost of the engine and the turbine side by side above the turbine's minimum size
    G0 = dict(M.GEN)
    M.GEN["TUR"] = (G0["TUR"][0], G0["TUR"][1], G0["TUR"][2], 1e12, 1e13)
    Be = M.route_B(Rs, Us, p, Ccf_s, Ocf_s)
    M.GEN.clear()
    M.GEN.update(G0)
    band = tur & (flows <= 2000.0)
    if band.any():
        mac("DTurNearPct", float(np.max(np.abs(Bs["lcoh"][band] / Be["lcoh"][band] - 1))) * 100, "{:.1f}")
    big = tur & (flows >= 5000.0)
    if big.any():
        mac("DTurBigPct", float(np.min(1 - Bs["lcoh"][big] / Be["lcoh"][big])) * 100, "{:.0f}")
    # the upgrading correlation around its lower bound: local cost elasticity on each side, and two single power laws in place of the exponential form
    C17 = 6.00e6 * np.exp(1.766e-4 * M.RNG_Q_LO)
    mac("DRngElastAbove", 1.766e-4 * M.RNG_Q_LO, "{:.2f}")
    mac("DRngElastOneQ", 1.0 / 1.766e-4, "{:.0f}")              # flow where the elasticity is one
    mac("DRngElastBelow", p["n_up"], "{:.2f}")
    qq = np.linspace(M.RNG_Q_LO, M.RNG_Q_HI, 200)
    nfit = float(np.polyfit(np.log(qq), np.log(6.00e6 * np.exp(1.766e-4 * qq)), 1)[0])
    mac("DRngFitExp", nfit, "{:.2f}")
    orig = M.upgrading_cost

    def single(n_):
        def f_(Q, fL, n_up):
            Q = np.clip(np.asarray(Q, float), 1e-6, None)
            C = C17 * (Q / M.RNG_Q_LO) ** (n_up if n_ is None else n_)
            return fL * C * M.ESC[2019], fL * (147.0 * Q + 1.48e5) * M.ESC[2019]
        return f_
    lc0, Qf = base["lcohA"], base["Qfeed"]
    inter = act & (Qf >= M.RNG_Q_LO) & (Qf <= M.RNG_Q_HI) & np.isfinite(lc0)
    above = act & (Qf > M.RNG_Q_HI) & np.isfinite(lc0)
    mac("DRngNinter", int(inter.sum()), "{:d}")
    mac("DRngNabove", int(above.sum()), "{:d}")
    for tag, n_ in (("A", None), ("B", nfit)):
        M.upgrading_cost = single(n_)
        alt = M.one_draw(p)
        M.upgrading_cost = orig
        d = 100 * (alt["lcohA"] / lc0 - 1)
        mac(f"DRng{tag}InterMax", float(np.nanmax(np.abs(d[inter]))), "{:.1f}")
        mac(f"DRng{tag}InterMed", float(np.nanmedian(d[inter])), "{:+.1f}")
        mac(f"DRng{tag}AboveMin", float(np.nanmin(d[above])) if above.any() else float("nan"), "{:+.1f}")
        mac(f"DRng{tag}AboveMax", float(np.nanmax(d[above])) if above.any() else float("nan"), "{:+.1f}")
        mac(f"DRng{tag}AboveAbs", float(np.nanmax(np.abs(d[above]))) if above.any() else float("nan"), "{:.1f}")
        if tag == "A":
            print("sites above the upgrading range:", [M.SITES[i] for i in np.flatnonzero(above)])
        small = act & (Qf < M.RNG_Q_LO) & np.isfinite(lc0) & (base["H2A"] > 1e3)
        mac(f"DRng{tag}SmallMax", float(np.nanmax(np.abs(d[small]))), "{:.1f}")
    # water: an on-site purification unit rejecting 40 % of its intake, at 5 kWh/m3 of intake and 10 USD/m3 of purified water, bounds the cost and energy that water adds
    vw = 12.0e-3                                  # m3/kg H2
    mac("DWaterCostHi", vw * 10.0, "{:.2f}")
    mac("DWaterCostBase", vw * 1.50, "{:.2f}")                   # at the price of Table 15
    mac("DWaterKwhHi", vw / 0.6 * 5.0, "{:.2f}")
    mac("DWaterKwhShare", 100 * vw / 0.6 * 5.0 / p["sec"], "{:.2f}")
    # compression of electrolytic hydrogen from 1 to 30 bar; E_COMP is this work at 25 C divided by a compressor efficiency of 0.6
    w_iso = 8.314 * 298.15 * np.log(30.0) / 2.016e-3 / 3.6e6       # kWh/kg, isothermal at 25 C
    if abs(w_iso / 0.6 - M.E_COMP) > 0.01:
        raise SystemExit(f"E_COMP {M.E_COMP} does not match the isothermal work {w_iso:.3f}/0.6")
    mac("DCompIso", w_iso, "{:.2f}")
    mac("DComp", M.E_COMP, "{:.1f}")
    mac("DCompPct", 100 * M.E_COMP / p["sec"], "{:.1f}")
    # Doña Juana: the stream already destroyed bought at its value as engine electricity
    j = M.DJ
    val_m3 = M.HHV_CH4 / M.GEN["ENG"][0] * M.GEN["ENG"][1] * p["pe"]     # USD per m3 CH4
    buy = base["eta0"][j] * base["gen"][j] * val_m3 / base["H2A"][j]
    mac("DBuyDJ", buy, "{:.2f}")
    # net electricity of a gas turbine relative to a standard engine burning the same gas
    net = lambda g: M.GEN[g][1] / M.GEN[g][0]
    mac("DTurNetLossPct", 100 * (1 - net("TUR") / net("ENG")), "{:.0f}")
    # share of the mass deposited in 2024 at the active sites that the diversion variant removes once it reaches its full level (the two fastest decay categories hold food and garden waste)
    j24 = int(np.flatnonzero(M.DEP == 2024)[0])
    org = M.Wc[M.active][:, :2, j24].sum() / M.W[M.active][:, j24].sum()
    mac("DDivMswPct", 100 * M.DIV_LEVEL * org, "{:.0f}")
    mac("DOrgSharePct", 100 * org, "{:.0f}")
    mac("DValGasEng", val_m3, "{:.2f}")
    mac("DEtaZeroShareDJ", 100 * base["eta0"][j] / base["eta"][j], "{:.0f}")
    # yield implied by the national estimate for the United States: 87 % of the methane purified and 3.3 kg CH4 per kg H2 (Saur and Milbrandt 2014)
    mac("DYieldNREL", 0.87 / 3.3 * M.RHO_CH4, "{:.3f}")
    pd.Series(out, name="value").rename_axis("name").to_csv(ROOT / "results" / "numbers_defaults.csv")
    for k, v in out.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
