"""National results and the named values that the text quotes.

Reads the model ensemble and the auxiliary outputs, writes
    results/national_results.json
    results/numbers_results.csv

Intervals are the 10th to 90th percentiles of the Monte Carlo ensemble unless the name of a value says otherwise. The results cover the sites active in 2024, the scope of the article; the inventory values describe the whole registry, and the closed sites enter only through their share of the potential. Run from the project root after the model, the validation and the sensitivity scripts:
    python scripts/17_aggregate.py
"""
import json
import re
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
GW_KT = 1e6 * 8760 * 0.5 / 53.0 / 1e6        # kt/yr from 1 GW of electrolysis, CF 0.5, 53 kWh/kg
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 30: "thirty"}


def q(a, p=(10, 50, 90)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, x)) for x in p]


def main():
    site = pd.read_csv(RES / "site_results.csv", index_col=0)
    nat = pd.read_csv(RES / "mc_national.csv")
    z = np.load(RES / "mc_site_draws.npz", allow_pickle=True)
    val = json.loads((RES / "validation.json").read_text())
    sens = pd.read_csv(RES / "sensitivity.csv")
    wchk = json.loads((RES / "waste_checks.json").read_text())
    capex = json.loads((RES / "capex_final.json").read_text())
    cst = json.loads((ROOT / "results" / "composition_stats.json").read_text())
    pstats = json.loads((RES / "precip_stats.json").read_text())
    meta = json.loads((RES / "model_meta.json").read_text())
    R, M = {}, {}

    def mac(name, value, fmt="{:.1f}"):
        M[name] = fmt.format(value) if not isinstance(value, str) else value

    def mac3(name, arr, fmt="{:.1f}"):
        lo, md, hi = arr
        mac(name, md, fmt)
        mac(name + "Lo", lo, fmt)
        mac(name + "Hi", hi, fmt)

    act = site["status_2024"] == "active"
    lic = site["licence_status"]
    mac("RNdraws", meta["ndraw"], "{:d}")
    mac("RSeed", meta["seed"], "{:d}")

    # ---------------------------------------------------------------- inventory
    R["n_sites"] = int(len(site))
    R["n_active"] = int(act.sum())
    R["n_closed"] = int((~act).sum())
    mac("RNsites", R["n_sites"], "{:d}")
    mac("RNactive", R["n_active"], "{:d}")
    mac("RNclosed", R["n_closed"], "{:d}")
    crosswalk = pd.read_csv(RES / "site_crosswalk.csv")
    mac("RNraw", crosswalk.groupby(["site_dane_code", "site_name"], dropna=False).ngroups, "{:d}")
    cls = site.loc[act, "site_type"].value_counts()
    for k, nm in [("Sanitary Landfill", "Landfill"), ("Open Dump", "Dump"),
                  ("Contingency Cell", "Contingency"), ("Transitional Cell", "Transitional")]:
        mac(f"RNactive{nm}", int(cls.get(k, 0)), "{:d}")
    mac("RNactiveOther", int(cls.sum() - sum(cls.get(k, 0) for k in
        ["Sanitary Landfill", "Open Dump", "Contingency Cell", "Transitional Cell"])), "{:d}")
    for g, nm in [("licensed", "Licensed"), ("expired", "Expired"), ("not reported", "NotRep")]:
        sel = act & (lic == g)
        mac(f"RN{nm}", int(sel.sum()), "{:d}")
        mac(f"RTons{nm}", site.loc[sel, "tons_2024"].sum() / 1e6, "{:.2f}")
    mac("RTonsActive", site.loc[act, "tons_2024"].sum() / 1e6, "{:.2f}")
    bins = site.loc[act, "remaining_life_bin"].value_counts()
    tb = site.loc[act].groupby("remaining_life_bin")["tons_2024"].sum() / 1e6
    for b, nm in [("0-2 yr", "LifeA"), ("2-5 yr", "LifeB"), ("5-10 yr", "LifeC"),
                  ("10-15 yr", "LifeD"), (">15 yr", "LifeE")]:
        mac(f"RN{nm}", int(bins.get(b, 0)), "{:d}")
        mac(f"RTons{nm}", float(tb.get(b, 0.0)), "{:.2f}")
    share_short = site.loc[act & ((lic == "expired") | (site["remaining_life_bin"] == "0-2 yr")),
                           "tons_2024"].sum() / site.loc[act, "tons_2024"].sum()
    mac("RShareShortLife", 100 * share_short, "{:.0f}")
    cl = site.loc[~act]
    for (a, b), nm in [((2011, 2014), "A"), ((2015, 2019), "B"), ((2020, 2024), "C")]:
        mac(f"RNclosed{nm}", int(cl["closure_year"].between(a, b).sum()), "{:d}")
    mac("RTonsClosed", cl["tons_2010_2024"].sum() / 1e6, "{:.1f}")
    mac("RTonsAll", site["tons_2010_2024"].sum() / 1e6, "{:.1f}")
    both = site["licence_end"].notna() & site["licence_end_inventory"].notna()
    mac("RNlicMatch", int(both.sum()), "{:d}")
    mac("RWasteTwentySix", wchk["series_total_Mt"]["2026"], "{:.2f}")
    mac("RWasteForty", wchk["series_total_Mt"]["2040"], "{:.2f}")
    mac("RWasteNinetyFive", wchk["series_total_Mt"]["1995"], "{:.2f}")
    mac("RPreOpening", wchk["pre_opening_pool_Mt_1995_2009"], "{:.1f}")
    mac("RGapFill", wchk["gap_fill_Mt_2010_2024"], "{:.2f}")
    mac("RBeyondDeposits", 100 * wchk["beyond_licence_share_2026_2040"], "{:.0f}")
    doc = site["opened_source"].fillna("").str.contains("CDM|SUI|Alcald") & act
    reg = pd.read_csv(ROOT / "inputs" / "landfilling_site_code_imputed.csv", dtype=str)
    reg["t"] = pd.to_numeric(reg["avg_tons_per_year"], errors="coerce")
    reg = reg[reg["t"] > 0]
    r24 = reg[reg["year"] == "2024"]
    mac("RShareSLmass", 100 * r24.loc[r24.site_type == "Sanitary Landfill", "t"].sum() / r24["t"].sum(), "{:.1f}")
    mac("RNrecords", len(reg), "{:d}")
    mac("RNmunicipalities", reg["served_dane_code"].nunique(), "{:d}")
    mac("RShareUnrecorded", 100 * reg["site_type"].isna().mean(), "{:.2f}")
    mac("RNopenDoc", int(doc.sum()), "{:d}")
    mac("RNopenCDM", int((site["opened_source"].fillna("").str.contains("CDM") & act).sum()), "{:d}")
    mac("RNzone", pstats["n_sites_assigned"], "{:d}")

    # ---------------------------------------------------------------- resource
    mac3("RGen", q(nat["gen_Mm3"]), "{:.0f}")
    mac3("RRec", q(nat["rec_Mm3"]), "{:.0f}")
    mac3("RGenKt", q(nat["gen_kt"]), "{:.0f}")
    mac3("RRecKt", q(nat["rec_kt"]), "{:.0f}")
    mac3("RGenTwentyFour", q(nat["gen2024_kt"]), "{:.0f}")
    mac3("RRecBeyond", [100 * x for x in q(nat["rec_beyond_share"])], "{:.0f}")
    # IPCC carbon-balance cross-check of the potential capacity, tonnage-weighted
    comp = pd.read_csv(ROOT / "results" / "composition_municipality.csv",
                       dtype={"municipality_code": str})
    doc_j = {"food": 0.15, "garden": 0.20, "paper": 0.40, "wood": 0.43, "textiles": 0.24,
             "diapers": 0.24}
    comp["DOC"] = sum(comp[k] / 100.0 * v for k, v in doc_j.items())
    comp["L0_ipcc"] = 1e3 / 0.678 * comp["DOC"] * 0.5 * 0.5 * 16.0 / 12.0
    wm = pd.read_csv(RES / "waste_municipality_site_year.csv", dtype={"served_dane_code": str})
    wm = wm[wm["site_id"].isin(site.index[act])]
    tm = wm[wm.year.between(2010, 2024)].groupby("served_dane_code")["tons"].sum()
    cm = comp.set_index(comp["municipality_code"].str.zfill(5))
    w_ = tm.reindex(cm.index).fillna(0.0)
    mac("RLzeroIPCC", float((cm["L0_ipcc"] * w_).sum() / w_.sum()), "{:.1f}")
    mac("RDOC", float((cm["DOC"] * w_).sum() / w_.sum()), "{:.3f}")
    l0i = float((cm["L0_ipcc"] * w_).sum() / w_.sum())
    # capacity and measured share weighted by the mass the active sites received, 2010-2024
    csa = pd.read_csv(ROOT / "results" / "composition_site.csv", encoding="utf-8-sig")
    csa = csa[csa["site_id"].isin(site.index[act])]
    l0a = float(np.average(csa["L0_m3CH4_per_Mg"], weights=csa["tons"]))
    sma = float(np.average(csa["share_measured"], weights=csa["tons"]))
    mac("RLzero", l0a, "{:.1f}")
    mac("RLzeroDiff", 100 * abs(l0a - l0i) / l0a, "{:.0f}")
    mac("RShareMeasured", sma, "{:.1f}")
    mac("RShareDept", 100.0 - sma, "{:.1f}")
    cmu = pd.read_csv(ROOT / "results" / "composition_municipality.csv")
    for tier, nm in (("measured", "Meas"), ("departmental", "Dept")):
        v = cmu.loc[cmu.tier == tier, "L0_m3CH4_per_Mg"]
        mac(f"RLzero{nm}Lo", v.min(), "{:.1f}")
        mac(f"RLzero{nm}Hi", v.max(), "{:.1f}")
    csite = pd.read_csv(ROOT / "results" / "composition_site.csv", encoding="utf-8-sig")
    csite = csite[csite["site_id"].isin(site.index[act])]
    mac("RLzeroSiteLo", csite["L0_m3CH4_per_Mg"].min(), "{:.1f}")
    mac("RLzeroSiteHi", csite["L0_m3CH4_per_Mg"].max(), "{:.1f}")
    exc = site["site_type"].isin(["Water Body", "Treatment Plant", "Open Burning"])
    mac("RTonsExcluded", site.loc[exc, "tons_2010_2024"].sum() / 1e3, "{:.0f}")
    mac("RNexcluded", int(exc.sum()), "{:d}")
    mac("RNslowMoved", int(cst.get("n_slow_moved", 0)), "{:d}")
    mac("RNappyMean", cst["nappy_mean_itemized"], "{:.2f}")

    # ---------------------------------------------------------------- scope
    # from here on every site-level count, share and ranking covers the active sites
    site_all = site
    site = site_all.loc[act].copy()
    sids_all = z["site_id"].astype(str)
    ja = site_all["status_2024"].reindex(sids_all).eq("active").to_numpy()
    zs = {k: z[k][:, ja] for k in ("H2A", "lcohA", "H2B", "lcohB", "H2A_cc", "lcohA_cc")}
    mac3("RHAall", q(nat["H2A_all_kt"]), "{:.1f}")
    mac3("RShareClosedA", [100 * x for x in q(nat["H2A_closed_kt"] / nat["H2A_all_kt"])], "{:.1f}")
    mac3("RShareActiveA", [100 * x for x in q(1 - nat["H2A_closed_kt"] / nat["H2A_all_kt"])], "{:.1f}")
    mac3("RShareClosedB", [100 * x for x in q(nat["H2B_closed_kt"] / nat["H2B_all_kt"])], "{:.1f}")
    mac("RShareClosedMaxA", 100 * float(np.max(nat["H2A_closed_kt"] / nat["H2A_all_kt"])), "{:.1f}")
    mac("RNclosedTen", int(((site_all["status_2024"] != "active") & (site_all["H2A_p50"] >= 1e4)).sum()), "{:d}")
    mac("RShareClosedGen", 100 * site_all.loc[~act, "gen_p50"].sum() / site_all["gen_p50"].sum(), "{:.1f}")

    # ---------------------------------------------------------------- hydrogen
    for k, nm in [("H2A_kt", "RHA"), ("H2A_lic_kt", "RHAlic"), ("H2A_singh_kt", "RHAsingh"),
                  ("H2A_cc_kt", "RHAcc"), ("H2B_kt", "RHB"), ("H2A_pot_kt", "RHApot"),
                  ("H2A_closed_kt", "RHAclosed"), ("H2A_licensed_kt", "RHAgLicensed"),
                  ("H2A_expired_kt", "RHAgExpired"), ("H2A_notreported_kt", "RHAgNotRep"),
                  ("H2A_lic_licensed_kt", "RHAlicLicensed"), ("H2A_lic_expired_kt", "RHAlicExpired")]:
        mac3(nm, q(nat[k]), "{:.1f}")
    R["H2A_kt"] = q(nat["H2A_kt"])
    mac3("RGWshareA", [100 * x / GW_KT for x in q(nat["H2A_kt"])], "{:.0f}")
    mac3("RGWshareAlic", [100 * x / GW_KT for x in q(nat["H2A_lic_kt"])], "{:.0f}")
    mac3("RGWshareB", [100 * x / GW_KT for x in q(nat["H2B_kt"])], "{:.0f}")
    mac("RGWkt", GW_KT, "{:.1f}")
    mac("RShareLicA", 100 * float(np.median(nat["H2A_lic_kt"] / nat["H2A_kt"])), "{:.0f}")
    mac("RRatioAB", float(np.median(nat["H2A_kt"] / nat["H2B_kt"])), "{:.1f}")

    # concentration, from the per-draw site potentials
    H = zs["H2A"].astype(float)
    Hs = -np.sort(-np.nan_to_num(H), axis=1)
    cum = np.cumsum(Hs, axis=1) / np.maximum(Hs.sum(axis=1, keepdims=True), 1e-9)
    for n in (5, 10, 30):
        mac3(f"RTop{WORDS.get(n, str(n)).capitalize()}", [100 * x for x in q(cum[:, n - 1])], "{:.0f}")
    n90 = (cum < 0.90).sum(axis=1) + 1
    mac3("RNninety", q(n90), "{:.0f}")
    sids = sids_all[ja]
    j = int(np.flatnonzero(sids == "11001|DONA JUANA")[0])
    mac3("RShareDJ", [100 * x for x in q(H[:, j] / H.sum(axis=1))], "{:.0f}")
    dept = site["dept"].reindex(sids).to_numpy()
    top4 = pd.Series(site.groupby("dept")["H2A_p50"].sum()).nlargest(4)
    R["top4_depts"] = list(top4.index)
    m4 = np.isin(dept, top4.index)
    mac3("RShareTopFourDept", [100 * x for x in q(H[:, m4].sum(axis=1) / H.sum(axis=1))], "{:.0f}")
    mac("RNpositive", int((site["H2A_p50"] > 1000).sum()), "{:d}")
    t30 = site.nlargest(30, "H2A_p50")
    mac("RShareDeptTopThirty", 100 * (1 - (t30["share_measured"] * t30["tons_2010_2024"]).sum()
                                        / t30["tons_2010_2024"].sum()), "{:.0f}")

    # route B reach and technology
    feas = site["B_feasible_share"] > 0.5
    mac3("RNBfeasible", q(nat["B_feasible_sites"]), "{:.0f}")
    mac("RNBfeasibleMed", int(feas.sum()), "{:d}")
    mac("RNBinfeasibleActive", int((~feas).sum()), "{:d}")
    tech = site.loc[feas, "techB_mode"].value_counts()
    for k, nm in [("MTUR", "Mtur"), ("SENG", "Seng"), ("ENG", "Eng"), ("TUR", "Tur")]:
        mac(f"RNtech{nm}", int(tech.get(k, 0)), "{:d}")
    pel = site.loc[feas, "Pel_p50"]
    mac("RPelMedian", float(pel.median()), "{:.0f}")
    mac("RPelNinety", float(pel.quantile(0.9)) / 1e3, "{:.1f}")
    mac3("RPelMW", q(nat["Pel_MW"]), "{:.0f}")
    mac("RTopSevenMWlo", min(val["top7_MW"]), "{:.1f}")
    mac("RTopSevenMWhi", max(val["top7_MW"]), "{:.1f}")
    mac("RDJmw", val["top7_MW"][0], "{:.1f}")
    techA = site.loc[site["H2A_p50"] > 0, "techA_mode"].value_counts()
    R["techA_mix"] = techA.to_dict()

    # ---------------------------------------------------------------- cost
    for thr in (2, 3, 4, 5):
        w = WORDS[thr].capitalize()
        mac3(f"RAbelow{w}", [100 * x for x in q(nat[f"A_below_{thr}"])], "{:.0f}")
        mac3(f"RBbelow{w}", [100 * x for x in q(nat[f"B_below_{thr}"])], "{:.0f}")
    lcc, Hcc = zs["lcohA_cc"].astype(float), zs["H2A_cc"].astype(float)
    for thr in (3, 4):
        s = np.where(np.isfinite(lcc) & (lcc < thr), Hcc, 0).sum(axis=1) / Hcc.sum(axis=1)
        mac3(f"RAccbelow{WORDS[thr].capitalize()}", [100 * x for x in q(s)], "{:.0f}")
    mac3("RLcohA", q(nat["lcohA_wmed"]), "{:.2f}")
    mac3("RLcohB", q(nat["lcohB_wmed"]), "{:.2f}")
    mac3("RLcohAcc", q(nat["lcohA_cc_wmed"]), "{:.2f}")
    for sid, nm in [("11001|DONA JUANA", "DJ"), ("05237|LA PRADERA", "Pradera"),
                    ("76890|COLOMBA EL GUABAL", "Colomba")]:
        r = site.loc[sid]
        mac(f"RLcohA{nm}", r["lcohA_p50"], "{:.2f}")
        mac(f"RLcohA{nm}Lo", r["lcohA_p10"], "{:.2f}")
        mac(f"RLcohA{nm}Hi", r["lcohA_p90"], "{:.2f}")
        mac(f"RLcohAcc{nm}", r["lcohA_cc_p50"], "{:.2f}")
        mac(f"RCcDelta{nm}", r["lcohA_cc_p50"] - r["lcohA_p50"], "{:.2f}")
        # cost of carbon dioxide avoided by capture, USD per tonne (IEAGHG definition)
        mac(f"RCcAvoidCost{nm}", 1000 * (r["lcohA_cc_p50"] - r["lcohA_p50"])
            / (r["ciA_p50"] - r["ciA_cc_p50"]), "{:.0f}")
        mac(f"RLcohB{nm}", r["lcohB_p50"], "{:.2f}")
        mac(f"RLcohAsingh{nm}", r["lcohA_singh_p50"], "{:.2f}")
    mac("RNbelowThree", int((site["lcohA_p50"] < 3).sum()), "{:d}")
    # market value of the electricity Route B consumes, per kg of hydrogen (same at every site)
    r = site.loc["11001|DONA JUANA"]
    mac("ROppB", r["oppB_p50"], "{:.1f}")
    mac("ROppBLo", r["oppB_p10"], "{:.1f}")
    mac("ROppBHi", r["oppB_p90"], "{:.1f}")
    # cost of carbon dioxide avoided at the three largest sites, lowest and highest
    ca = [1000 * (site.loc[k, "lcohA_cc_p50"] - site.loc[k, "lcohA_p50"])
          / (site.loc[k, "ciA_p50"] - site.loc[k, "ciA_cc_p50"])
          for k in ("11001|DONA JUANA", "05237|LA PRADERA", "76890|COLOMBA EL GUABAL")]
    mac("RCcAvoidCostLo", min(ca), "{:.0f}")
    mac("RCcAvoidCostHi", max(ca), "{:.0f}")
    # route ranking at the conservative bound of the reformer efficiency (site medians)
    # route ranking at the conservative bound of the reformer efficiency, compared draw by draw in the draws where both routes can be built, at sites where Route B is feasible in most draws
    cross = feas & (site["routeA_singh_share"] <= 0.5)
    mac("RNcrossSingh", int(cross.sum()), "{:d}")
    mac("RNcrossSinghActive", int((cross & (site["status_2024"] == "active")).sum()), "{:d}")
    mac("RShareCrossSingh", 100 * site.loc[cross, "H2A_singh_p50"].sum() / site["H2A_singh_p50"].sum(),
        "{:.1f}")
    lowest = np.minimum(site["lcohA_singh_p50"], site["lcohB_p50"])
    mac("RCrossSinghMinCost", float(lowest[cross].min()) if cross.any() else float("nan"), "{:.1f}")
    # share of the extension-dependent potential (site medians) held by the two largest sites
    ext = (site["H2A_p50"] - site["H2A_lic_p50"]).clip(lower=0)
    mac("RShareExtTopTwo", 100 * ext[["11001|DONA JUANA", "05237|LA PRADERA"]].sum() / ext.sum(),
        "{:.0f}")
    mac("RNbelowTwo", int((site["lcohA_p50"] < 2).sum()), "{:d}")
    both_ = feas & np.isfinite(site["lcohB_p50"])
    bw = both_ & (site["routeA_share"] <= 0.5)
    mac("RNAwins", int((both_ & (site["routeA_share"] > 0.5)).sum()), "{:d}")
    mac("RNBwins", int(bw.sum()), "{:d}")
    mac("RBwinsMinCost", float(np.minimum(site["lcohA_p50"], site["lcohB_p50"])[bw].min())
        if bw.any() else float("nan"), "{:.0f}")
    mac("RBwinsMaxH", float(site.loc[bw, "H2A_p50"].max() / 1e3) if bw.any() else 0.0, "{:.0f}")
    mac("RShareBwins", 100 * site.loc[bw, "H2A_p50"].sum() / site["H2A_p50"].sum(), "{:.1f}")
    # recovered gas at the sites where Route B cannot be built in most draws
    mac("RShareRecBinfeasible", 100 * site.loc[~feas, "rec_p50"].sum() / site["rec_p50"].sum(), "{:.1f}")
    # sites whose reformer lies below the smallest plant of the cost record (100 kg/d)
    small = (site["S_p50"] < 100) & (site["H2A_p50"] > 0)
    mac("RNsmallS", int(small.sum()), "{:d}")
    mac("RShareSmallS", 100 * site.loc[small, "H2A_p50"].sum() / site["H2A_p50"].sum(), "{:.1f}")
    # sites above the cost axis of Figure 5 (cut at 40 USD/kg; sites below 1 t/yr not drawn)
    drawn = site["H2A_p50"] > 1e3
    off = drawn & (site["lcohA_p50"] > 40)
    mac("RNfigFiveOff", int(off.sum()), "{:d}")
    mac("RNfigFiveDrawn", int(drawn.sum()), "{:d}")
    mac("RShareFigFiveOff", 100 * site.loc[off, "H2A_p50"].sum() / site["H2A_p50"].sum(), "{:.1f}")
    # hydrogen available below 3 USD/kg, kt/yr
    mac3("RAbelowThreeKt", q(nat["A_below_3"] * nat["H2A_kt"]), "{:.0f}")
    # the reformer capital drawn as a common level plus an independent residual per site
    fsplit = RES / "mc_national_split.csv"
    if fsplit.exists():
        ns = pd.read_csv(fsplit)
        mac3("RAbelowThreeSplit", [100 * x for x in q(ns["A_below_3"])], "{:.0f}")
        mac3("RLcohASplit", q(ns["lcohA_wmed"]), "{:.2f}")
        mac3("RHASplit", q(ns["H2A_kt"]), "{:.1f}")
    # the national potential in the lowest and highest tenth of the collection efficiency
    par = pd.read_csv(RES / "mc_params.csv")
    lo_d, hi_d = par["u_eta"] <= par["u_eta"].quantile(0.1), par["u_eta"] >= par["u_eta"].quantile(0.9)
    mac("RHAetaLo", float(nat.loc[lo_d, "H2A_kt"].median()), "{:.0f}")
    mac("RHAetaHi", float(nat.loc[hi_d, "H2A_kt"].median()), "{:.0f}")
    # potential that depends on extensions at the two largest sites, kt/yr
    mac("RExtTopTwoKt", float(ext[["11001|DONA JUANA", "05237|LA PRADERA"]].sum() / 1e6), "{:.0f}")
    mac3("RAcheaper", [100 * x for x in q(nat["A_cheaper_share"])], "{:.0f}")

    # ---------------------------------------------------------------- variants
    # gas already destroyed at four sites left to the existing system
    mac3("RHAincr", q(nat["H2A_incr_kt"]), "{:.1f}")
    mac("RShareIncrLoss", 100 * float(np.median(1 - nat["H2A_incr_kt"] / nat["H2A_kt"])), "{:.0f}")
    mac3("RAbelowThreeIncr", [100 * x for x in q(nat["A_below_3_incr"])], "{:.0f}")
    mac3("RLcohAincr", q(nat["lcohA_incr_wmed"]), "{:.2f}")
    rdj = site.loc["11001|DONA JUANA"]
    mac("RHAincrDJ", rdj["H2A_incr_p50"] / 1e6, "{:.1f}")
    mac("RHADJ", rdj["H2A_p50"] / 1e6, "{:.1f}")
    mac("RLcohAincrDJ", rdj["lcohA_incr_p50"], "{:.2f}")
    # half of the food and garden waste diverted by 2035, phased in from 2026
    mac3("RHAdiv", q(nat["H2A_div_kt"]), "{:.1f}")
    mac3("RHAlicdiv", q(nat["H2A_lic_div_kt"]), "{:.1f}")
    mac("RShareDivLoss", 100 * float(np.median(1 - nat["H2A_div_kt"] / nat["H2A_kt"])), "{:.0f}")
    mac("RShareLicDivLoss", 100 * float(np.median(1 - nat["H2A_lic_div_kt"] / nat["H2A_lic_kt"])), "{:.0f}")
    mac("RGenDivLoss", 100 * float(np.median(1 - nat["gen_div_Mm3"] / nat["gen_Mm3"])), "{:.0f}")
    mac3("RAbelowThreeDiv", [100 * x for x in q(nat["A_below_3_div"])], "{:.0f}")
    t30d = site.nlargest(30, "H2A_p50")
    mac("RShareDivLossTopThirty", 100 * (1 - t30d["H2A_div_p50"].sum() / t30d["H2A_p50"].sum()), "{:.0f}")
    # electrolytic hydrogen compressed to the pressure of the reformer
    mac3("RHBcomp", q(nat["H2B_comp_kt"]), "{:.1f}")
    mac("RShareCompLoss", 100 * float(np.median(1 - nat["H2B_comp_kt"] / nat["H2B_kt"])), "{:.1f}")
    mac("RLcohBcompRise", 100 * float(np.median(nat["lcohB_comp_wmed"] / nat["lcohB_wmed"] - 1)), "{:.1f}")
    # the generator switch of Route B at the active sites (modal choice across draws)
    fz = site["B_feasible_share"] > 0.5
    rec_h = site["rec_p50"] / 8760.0
    turs, engs = fz & (site["techB_mode"] == "TUR"), fz & (site["techB_mode"] == "ENG")
    if turs.any():
        mac("RTurRecMin", float(rec_h[turs].min()), "{:.0f}")
        mac("RTurMWmin", float(rec_h[turs].min()) * 37.7 / (13.7 * 0.93) / 1e3, "{:.1f}")
    if engs.any():
        mac("REngRecMax", float(rec_h[engs].max()), "{:.0f}")
        mac("REngMWmax", float(rec_h[engs].max()) * 37.7 / (13.7 * 0.93) / 1e3, "{:.1f}")
    # engine sites whose recovered methane could fill the smallest gas turbine (defaults.py)
    dnum = pd.read_csv(RES / "numbers_defaults.csv", index_col="name")["value"]
    q_tur = float(dnum["DTurQ"])
    mac("RNengAboveTurQ", int((engs & act & (rec_h >= q_tur)).sum()), "{:d}")
    # share of draws in which the two largest sites reach the national cost objective of 1.7 USD/kg
    ids = list(z["site_id"])
    for sid, nm in (("11001|DONA JUANA", "DJ"), ("05237|LA PRADERA", "Pradera")):
        lc = np.asarray(z["lcohA"][:, ids.index(sid)], dtype=float)
        mac(f"R{nm}belowObjPct", 100 * float(np.mean(lc < 1.7)), "{:.0f}")

    # ---------------------------------------------------------------- carbon
    mac3("RCiA", q(nat["ciA_w"]), "{:.2f}")
    mac3("RCiAcc", q(nat["ciA_cc_w"]), "{:.1f}")
    mac3("RCiB", q(nat["ciB_w"]), "{:.1f}")
    mac3("RCiAmax", q(nat["ciA_max"]), "{:.2f}")
    mac3("RAvoid", q(nat["avoidA_w"]), "{:.0f}")
    mac3("RAvoidMt", q(nat["avoid_Mt"]), "{:.1f}")
    mac3("RCOtwoStored", q(nat["co2_stored_Mt"]), "{:.2f}")
    ciB_eng = site.loc[feas & site["techB_mode"].isin(["ENG", "SENG"]), "ciB_p50"]
    ciB_tur = site.loc[feas & (site["techB_mode"] == "TUR"), "ciB_p50"]
    ciB_mt = site.loc[feas & (site["techB_mode"] == "MTUR"), "ciB_p50"]
    mac("RCiBeng", float(ciB_eng.median()) if len(ciB_eng) else np.nan, "{:.1f}")
    mac("RCiBtur", float(ciB_tur.median()) if len(ciB_tur) else np.nan, "{:.2f}")
    mac("RCiBmtur", float(ciB_mt.median()) if len(ciB_mt) else np.nan, "{:.1f}")
    for k, g in (("ARfive", 28.0), ("ARfiveFb", 34.0)):
        mac(f"RCiA{k}", float(np.median(nat["ciA_w"])) * g / 27.0, "{:.2f}")
        mac(f"RCiB{k}", float(np.median(nat["ciB_w"])) * g / 27.0, "{:.1f}")
        mac(f"RAvoid{k}", float(np.median(nat["avoidA_w"])) * g / 27.0, "{:.0f}")
    mac("RAvoidDJ", site.loc["11001|DONA JUANA", "avoidA_p50"], "{:.0f}")

    # ---------------------------------------------------------------- validation
    mac3("RValDJgen", val["dj_gen_model_kt"], "{:.0f}")
    mac3("RValDJeff", [100 * x for x in val["dj_field_eff"]], "{:.0f}")
    mac3("RValDJeffAlt", [100 * x for x in val["dj_field_eff_alt"]], "{:.0f}")
    mac("RValDJdestroyed", val["dj_destroyed_kt"], "{:.1f}")
    # generation that the metered flow would imply at the lowest in-situ efficiency (daily cover)
    g41 = val["dj_destroyed_kt"] / 0.41
    mac("RValDJgenAtDaily", g41, "{:.0f}")
    mac("RValDJgenAtDailyPct", 100 * (1 - g41 / val["dj_gen_model_kt"][1]), "{:.0f}")
    mac("RValSSPDmean", val["registry_vs_sspd_mean_abs_pct"], "{:.1f}")
    mac("RValSSPDmax", max(abs(v) for v in val["registry_vs_sspd_pct"].values()), "{:.1f}")
    mac3("RValPlant", val["plant700_lcohA"], "{:.1f}")
    mac("RValPlantE", val["plant700_E"][1], "{:.2f}")
    mac("RValPlantKgd", val["plant700_H2_kg_d"], "{:.0f}")
    for key, nm in (("eta0_pradera", "Pradera"), ("eta0_presidente", "Presidente"),
                    ("eta0_dj", "DJ"), ("eta0_mondonedo", "Mondonedo")):
        mac(f"REtaZero{nm}", 100 * val[key][1], "{:.0f}")

    # ---------------------------------------------------------------- sensitivity
    for out, nm in (("H2A_kt", "HA"), ("lcohA_wmed", "LcohA"), ("A_below_3", "AbelowThree"),
                    ("ciA_w", "CiA"), ("ciB_w", "CiB"), ("lcohB_wmed", "LcohB")):
        t = sens[sens.output == out].reset_index(drop=True)
        mac(f"RSens{nm}Rtwo", t["r2_rank"].iloc[0], "{:.2f}")
        for r_ in range(3):
            lab = t["label"].iloc[r_]
            lab = lab if lab[:2].isupper() else lab[0].lower() + lab[1:]
            mac(f"RSens{nm}{['First', 'Second', 'Third'][r_]}", lab, "{}")
            mac(f"RSens{nm}{['First', 'Second', 'Third'][r_]}P", t["prcc"].iloc[r_], "{:.2f}")

    # ---------------------------------------------------------------- reformer efficiency
    E = site.loc[site["H2A_p50"] > 0]
    top = E.nlargest(30, "H2A_p50")
    mac("REtop", float(top["E_p50"].min()), "{:.2f}")
    mac("REtopHi", float(top["E_p50"].max()), "{:.2f}")
    mac("RSDJ", site.loc["11001|DONA JUANA", "S_p50"] / 1e3, "{:.0f}")
    R["numbers"] = M
    (RES / "national_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    pd.Series(M, name="value").rename_axis("name").to_csv(RES / "numbers_results.csv")
    print(f"{len(M)} values written")
    for k in list(M)[:400]:
        print(f"{k:28s} {M[k]}")


if __name__ == "__main__":
    main()
