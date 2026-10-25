"""Variants of the model, with the realizations of the base case.

Every realization takes the parameters of the base case from the same streams (same seed), so that each variant differs from the base case only in what it changes. The random numbers that only the variants use come from a separate stream. Each variant writes its national aggregates over the active sites, per realization, to results/variants_national.csv.

Variants
    gen78        generation 22 % lower at every site, the bias implied by the in situ field efficiency under daily cover at Doña Juana (Section 4.9)
    fmin         minimum load of 50 % of the design flow for the reformer train and for the generator of Route B; years below it produce nothing (fmin30: 30 % and 25 %)
    p75, max     design flow at the 75th percentile and at the maximum of the annual flows
    mod2         two identical reformer and upgrading trains, each sized at half of the maximum annual flow and with the minimum load of fmin
    etaind       collection efficiency with an independent quantile at each site
    etarho       collection efficiency with a correlation of 0.5 between sites (Gaussian copula)
    compdept     composition error common to the sites of each department
    capexpred    reformer CAPEX from the predictive distribution of the fit, common to all sites
    capexmix     reformer CAPEX from a mixed model with a random intercept by study
    dyn          collection efficiency of each annual deposit by cover stage (Section 3.7)
    opp          recovered methane priced at its netback as electricity (both routes)
    ccs          capture fraction 0.85 and 0.95, without CO2 transport and storage emissions, and with 1 % and 10 % of the stored CO2 lost
    rngx         operating cost of the cleanup and upgrading train x1.5 and x2, auxiliary electricity of that train x1.5
    techfix      Route B with the generator fixed at the modal technology of each site

Run from the project root:
    python scripts/20_variants.py [NDRAW]
"""
import importlib
import json
import pathlib
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
NDRAW = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10000)
sys.argv = [sys.argv[0], str(NDRAW)]
M = importlib.import_module("11_lfg_h2_model")

OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG_VAR = np.random.default_rng(M.SEED + 4)

GEN_SCALE = 0.78
# minimum stable load of a train: reformer and upgrading plants are usually designed to run continuously down to half of their design flow (30 % with design changes); a landfill-gas engine or turbine is held at half load, since efficiency and emissions worsen below it
FMIN_A, FMIN_B = 0.50, 0.50
FMIN_A_LO, FMIN_B_LO = 0.30, 0.25
RHO_ETA = 0.5
CAPT_LO, CAPT_HI = 0.85, 0.95
LEAK_LO, LEAK_HI = 0.01, 0.10   # share of the stored CO2 lost from storage (1 % in 100 yr; stress)
# collection efficiency of an annual deposit by its age and cover stage, sanitary landfills: no active collection during the first two years after placement, daily cover with active collection from the second year, intermediate cover from the fifth and long-term or final cover from the fifteenth, and final cover for every deposit from the year after the last placement (closure). Triangular, with the common quantile of the base case.
DYN_DAILY = (0.41, 0.50, 0.60)
DYN_INTER = (0.60, 0.69, 0.75)
DYN_FINAL = (0.71, 0.80, 0.85)
DYN_AGE = (2, 5, 15)            # first year of daily, intermediate and final cover
DYN_CAP = 1                     # years from the last placement to final cover of the site

act = M.active
NS = M.NS
lic = M.cat["licence_status"].to_numpy()
dept = M.cat["dept"].astype(str).to_numpy()
DEPTS, dept_idx = np.unique(dept, return_inverse=True)
# effective closure year of the licensed-only variant: first year with a deposit beyond the licensed end of life
closure = np.full(NS, 9999)
has_b = (M.beyond > 0).any(axis=1)
closure[has_b] = M.DEP[np.argmax(M.beyond[has_b] > 0, axis=1)]
site_res = pd.read_csv(ROOT / "results" / "site_results.csv", index_col=0)
MIX = json.loads((OUT / "capex_checks.json").read_text())["mixed"]
TECHB_MODE = np.array([M.GEN_NAMES.index(t) if t in M.GEN_NAMES else -1
                       for t in site_res["techB_mode"].reindex(M.SITES).fillna("none")])


def wmedian(x, w):
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    o = np.argsort(x[ok])
    cw = np.cumsum(w[ok][o])
    return float(x[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])


def kz_l0(p):
    kz = np.empty((4, 5))
    for c in range(4):
        for z in range(5):
            kz[c, z] = M.tri_ppf(p["u_k"][c], M.K[c, min(z + 1, 4)], M.K[c, z], M.K[c, max(z - 1, 0)])
    return kz, M.L0C * p["l0m"]


def rmul_of(z_comp):
    tot = np.maximum(M.W.sum(axis=1), 1e-9)
    wbar = np.clip(M.Wc.sum(axis=2) / tot[:, None], 1e-6, None)
    rest = np.clip(1.0 - wbar.sum(axis=1, keepdims=True), 1e-6, None)
    sd = np.where(M.share_measured[:, None] > 0.5, 0.10, 0.20)
    e = np.exp(np.log(wbar / rest) + sd * z_comp)
    return (e / (1.0 + e.sum(axis=1, keepdims=True))) / wbar


def classes(p):
    mcf = M.mcf0.copy()
    lo, md, hi = M.eta_lo.copy(), M.eta_md.copy(), M.eta_hi.copy()
    san = M.sanitary.copy()
    if M.unrec.any():
        names = M.mass_share.index.to_numpy()
        cum = np.cumsum(M.mass_share.to_numpy())
        for j in np.flatnonzero(M.unrec):
            c = names[min(np.searchsorted(cum, p["u_unrec"][j]), len(names) - 1)]
            mcf[j], lo[j], md[j], hi[j] = M.SITE_CLASS[c]
            san[j] = c == "Sanitary Landfill"
    return mcf, lo, md, hi, san


def generation(p, z_comp=None, dyn=None):
    """G and Gb of one_draw (all evaluated years); with dyn, the collected methane by cover stage, R_dyn of the base case and of the licensed-only case (2026-2040)."""
    kz, l0 = kz_l0(p)
    rmul = rmul_of(p["z_comp"] if z_comp is None else z_comp)
    mcf = classes(p)[0]
    G = np.zeros((NS, len(M.EVAL)))
    Gb = np.zeros((NS, len(M.EVAL)))
    if dyn is not None:
        Hb_, groups = dyn
        Rd_ = np.zeros((NS, len(M.EVAL)))
        Rl_ = np.zeros((NS, len(M.EVAL)))
    for z in range(5):
        sel = M.zone == z
        if not sel.any():
            continue
        for c in range(4):
            D = M.decay_kernel(kz[c, z])
            coef = kz[c, z] * l0[c] * rmul[sel, c]
            G[sel] += coef[:, None] * (M.Wc[sel, c, :] @ D)
            Gb[sel] += coef[:, None] * (M.Wc_bey[sel, c, :] @ D)
            if dyn is not None:
                Rd_[sel] += coef[:, None] * (M.Wc[sel, c, :] @ (D * Hb_))
                for y, Hy in groups.items():
                    s2 = sel & (closure == y)
                    if s2.any():
                        cf = kz[c, z] * l0[c] * rmul[s2, c]
                        Rl_[s2] += cf[:, None] * ((M.Wc[s2, c, :] - M.Wc_bey[s2, c, :]) @ (D * Hy))
    G *= mcf[:, None]
    Gb *= mcf[:, None]
    if dyn is None:
        return G, Gb
    return G, Gb, Rd_ * mcf[:, None], Rl_ * mcf[:, None]


def dyn_matrices(u):
    """Collection efficiency by deposit year and evaluated year: base case, and one matrix per closure year for the licensed-only case."""
    d, it, fi = (M.tri_ppf(u, *t) for t in (DYN_DAILY, DYN_INTER, DYN_FINAL))
    a = M.AGE
    Hb_ = np.where(a < DYN_AGE[0], 0.0, np.where(a < DYN_AGE[1], d, np.where(a < DYN_AGE[2], it, fi)))
    groups = {}
    for y in np.unique(closure):
        cap = M.EVAL[None, :] >= (y + DYN_CAP)
        groups[int(y)] = np.where(cap & (a >= 0), fi, Hb_)
    return Hb_, groups, (float(d), float(it), float(fi))


def eta0_of(p, G, eta_c):
    eta0 = np.zeros(NS)
    for j in np.flatnonzero(M.has_t | M.has_f):
        if M.has_t[j]:
            y0, y1 = M.cap_years[j]
            gy = G[j, (M.EVAL >= y0) & (M.EVAL <= y1)].mean()
            t = M.tri_ppf(p["u_cap"][j], *M.cap_t[j])
            eta0[j] = t * 1000.0 / M.RHO_CH4 / max(gy, 1e-9)
        else:
            eta0[j] = M.tri_ppf(p["u_cap"][j], *M.cap_f[j])
    return np.minimum(eta0, eta_c)


def cf_cost(p, Rd, eta_c, eta0):
    Qlfg = Rd / p["F"] / M.H
    incr = np.where(eta_c > 0, 1.0 - eta0 / np.where(eta_c > 0, eta_c, 1.0), 1.0)
    Ccf = incr * p["fL"] * (2.00e4 + M.Nw * (279.0 * (M.DEPTH - 3.05) + 1.77e4)
                            + 3329.0 * np.clip(Qlfg, 0, None) ** 0.61) * M.ESC[2013]
    Ocf = p["fL"] * (2600.0 * M.Nw + 5100.0) * M.ESC[2013]
    ok = Rd > 0
    return np.where(ok, Ccf, 0.0), np.where(ok, Ocf, 0.0)


def netback(p, Rd, U):
    """Value of the recovered methane as electricity sold at p_e, net of the levelized cost of the least-cost generator able to burn it (C&F excluded, since both uses need it), USD per m3 CH4, never negative."""
    i = p["i"]
    df = (1.0 + i) ** -M.DF_T
    best = np.zeros(Rd.shape)
    thermal_d = Rd * M.HHV_CH4
    for name in M.GEN_NAMES:
        HR, epar, CF, Pmin, Pmax = M.GEN[name]
        P = thermal_d / (HR * M.H * CF)
        feas = (P >= Pmin) & (P <= Pmax)
        if not feas.any():
            continue
        Eg = U * M.HHV_CH4 / HR
        Cg, _ = M.gen_cost(name, P, Eg.mean(axis=1))
        _, Og = M.gen_cost(name, np.broadcast_to(P[:, None], U.shape), Eg)
        Cg = p["fL"] * Cg * M.ESC[M.GEN_BASE[name]]
        Og = p["fL"] * Og * M.ESC[M.GEN_BASE[name]]
        cost = Cg + (Og * df).sum(axis=1)
        if name == "MTUR":
            cost = cost + Cg * (1.0 + i) ** -M.MTUR_LIFE
        rev = (p["pe"] * Eg * epar * df).sum(axis=1)
        gas = (U * df).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            nb = np.where(feas & (gas > 0), (rev - cost) / gas, 0.0)
        best = np.maximum(best, nb)
    return best


def disc_h2(H2_y, i):
    return (H2_y * (1.0 + i) ** -M.DF_T).sum(axis=1)


def metrics(tag, A=None, Al=None, B=None, i=None, n=None, A_cc=None):
    """National aggregates over the active sites."""
    if A is not None:
        H = A["H2_y"].mean(axis=1)[act]
        lc = np.where(np.isfinite(A["lcoh"]), A["lcoh"], np.nan)[act]
        n[f"{tag}_H2A_kt"] = H.sum() / 1e6
        n[f"{tag}_lcohA_wmed"] = wmedian(lc, H)
        n[f"{tag}_A_below_3"] = float(H[np.nan_to_num(lc, nan=np.inf) < 3.0].sum() / max(H.sum(), 1e-9))
        dh = disc_h2(A["H2_y"], i)[act]
        fin = np.isfinite(lc) & (dh > 0)
        n[f"{tag}_lcohA_port"] = float(np.sum(lc[fin] * dh[fin]) / dh[fin].sum())
        # the same aggregate over the 30 largest sites of the realization
        top = np.zeros(H.shape, dtype=bool)
        top[np.argsort(-H)[:30]] = True
        f30 = fin & top
        n[f"{tag}_lcohA_port30"] = float(np.sum(lc[f30] * dh[f30]) / max(dh[f30].sum(), 1e-9))
        n[f"{tag}_H2A_top30_kt"] = H[top].sum() / 1e6
        w = H > 0
        n[f"{tag}_ciA_w"] = float(np.nansum(A["ci"][act][w] * H[w]) / max(H[w].sum(), 1e-9))
    if Al is not None:
        n[f"{tag}_H2Alic_kt"] = Al["H2_y"].mean(axis=1)[act].sum() / 1e6
        Hl = Al["H2_y"].mean(axis=1)[act]
        n[f"{tag}_lcohAlic_wmed"] = wmedian(np.where(np.isfinite(Al["lcoh"]), Al["lcoh"], np.nan)[act], Hl)
    if B is not None:
        HB = B["H2_y"].mean(axis=1)[act]
        lB = B["lcoh"][act]
        n[f"{tag}_H2B_kt"] = HB.sum() / 1e6
        n[f"{tag}_lcohB_wmed"] = wmedian(lB, HB)
        dh = disc_h2(B["H2_y"], i)[act]
        fin = np.isfinite(lB) & (dh > 0)
        n[f"{tag}_lcohB_port"] = float(np.sum(lB[fin] * dh[fin]) / max(dh[fin].sum(), 1e-9))
        wb = HB > 0
        n[f"{tag}_ciB_w"] = float(np.nansum(B["ci"][act][wb] * HB[wb]) / max(HB[wb].sum(), 1e-9))
        if A is not None:
            H = A["H2_y"].mean(axis=1)[act]
            lA = A["lcoh"][act]
            both = np.isfinite(lA) & np.isfinite(lB)
            n[f"{tag}_A_cheaper_share"] = float(H[both & (lA < lB)].sum() / max(H[both].sum(), 1e-9))
    if A_cc is not None:
        Hc = A_cc["H2_y"].mean(axis=1)[act]
        w = Hc > 0
        n[f"{tag}_ciAcc_w"] = float(np.nansum(A_cc["ci"][act][w] * Hc[w]) / max(Hc[w].sum(), 1e-9))
        n[f"{tag}_lcohAcc_wmed"] = wmedian(A_cc["lcoh"][act], Hc)
        n[f"{tag}_co2_Mt"] = float(np.nansum(A_cc["co2"][act])) / 1e9


def main():
    rows, years = [], []
    t0 = time.time()
    for d in range(NDRAW):
        p = M.draw_params()
        # random numbers used only by the variants, drawn in a fixed order every realization
        u_ind = RNG_VAR.random(NS)
        eps = RNG_VAR.standard_normal(NS)
        z_dept = RNG_VAR.standard_normal((len(DEPTS), 4))
        z_pred = float(RNG_VAR.standard_normal())
        n = {}
        i = p["i"]
        # ---------------- base case, as one_draw
        G, Gb = generation(p)
        mcf, lo, md, hi, san = classes(p)
        eta_c = M.tri_ppf(p["u_eta"], lo, md, hi)
        eta0 = eta0_of(p, G, eta_c)
        Gh, Gbh = G[:, M.IH:], Gb[:, M.IH:]
        R = eta_c[:, None] * Gh
        Rl = eta_c[:, None] * (Gh - Gbh)
        Rd, Rld = R.mean(axis=1), Rl.mean(axis=1)
        U, Ul = np.minimum(R, Rd[:, None]), np.minimum(Rl, Rld[:, None])
        Ccf, Ocf = cf_cost(p, Rd, eta_c, eta0)
        ok = Rd > 0
        A = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf)
        A["lcoh"] = np.where(ok, A["lcoh"], np.nan)
        Al = M.route_A(Rl, Rld, Ul, p, "scale", False, Ccf, Ocf)
        Al["lcoh"] = np.where(ok, Al["lcoh"], np.nan)
        B = M.route_B(Rd, U, p, Ccf, Ocf)
        Acc = M.route_A(R, Rd, U, p, "scale", True, Ccf, Ocf)
        Acc["lcoh"] = np.where(ok, Acc["lcoh"], np.nan)
        metrics("base", A, Al, B, i, n, Acc)
        # national hydrogen by year, base and licensed-only
        yr = {"base": A["H2_y"][act].sum(axis=0), "lic": Al["H2_y"][act].sum(axis=0),
              "rec": R[act].sum(axis=0), "used": U[act].sum(axis=0),
              "rec_lic": Rl[act].sum(axis=0), "used_lic": Ul[act].sum(axis=0)}
        # ---------------- generation 22 % lower
        Rg, Rlg = GEN_SCALE * R, GEN_SCALE * Rl
        Rdg, Rldg = Rg.mean(axis=1), Rlg.mean(axis=1)
        Ccfg, Ocfg = cf_cost(p, Rdg, eta_c, np.minimum(eta0 / GEN_SCALE, eta_c))
        Ag = M.route_A(Rg, Rdg, np.minimum(Rg, Rdg[:, None]), p, "scale", False, Ccfg, Ocfg)
        Alg = M.route_A(Rlg, Rldg, np.minimum(Rlg, Rldg[:, None]), p, "scale", False, Ccfg, Ocfg)
        Bg = M.route_B(Rdg, np.minimum(Rg, Rdg[:, None]), p, Ccfg, Ocfg)
        metrics("gen78", Ag, Alg, Bg, i, n)
        # ---------------- minimum load
        Af = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf, {"fmin": FMIN_A})
        Alf = M.route_A(Rl, Rld, Ul, p, "scale", False, Ccf, Ocf, {"fmin": FMIN_A})
        Bf = M.route_B(Rd, U, p, Ccf, Ocf, {"fmin": FMIN_B})
        metrics("fmin", Af, Alf, Bf, i, n)
        Af = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf, {"fmin": FMIN_A_LO})
        Alf = M.route_A(Rl, Rld, Ul, p, "scale", False, Ccf, Ocf, {"fmin": FMIN_A_LO})
        Bf = M.route_B(Rd, U, p, Ccf, Ocf, {"fmin": FMIN_B_LO})
        metrics("fmin30", Af, Alf, Bf, i, n)
        # ---------------- design flow at the 75th percentile and at the maximum, two trains
        for tag, fn in (("p75", lambda x: np.percentile(x, 75, axis=1)), ("max", lambda x: x.max(axis=1))):
            Rdx, Rldx = fn(R), fn(Rl)
            Ccfx, Ocfx = cf_cost(p, Rdx, eta_c, eta0)
            Ax = M.route_A(R, Rdx, np.minimum(R, Rdx[:, None]), p, "scale", False, Ccfx, Ocfx,
                           {"fmin": FMIN_A})
            Alx = M.route_A(Rl, Rldx, np.minimum(Rl, Rldx[:, None]), p, "scale", False, Ccfx, Ocfx,
                            {"fmin": FMIN_A})
            metrics(tag, Ax, Alx, None, i, n)
        Rdx, Rldx = R.max(axis=1), Rl.max(axis=1)
        Ccfx, Ocfx = cf_cost(p, Rdx, eta_c, eta0)
        Am = M.route_A(R, Rdx, R, p, "scale", False, Ccfx, Ocfx, {"fmin": FMIN_A, "trains": 2})
        Alm = M.route_A(Rl, Rldx, Rl, p, "scale", False, Ccfx, Ocfx, {"fmin": FMIN_A, "trains": 2})
        metrics("mod2", Am, Alm, None, i, n)
        # ---------------- correlation of the collection efficiency between sites
        z_eta = float(ndtri(min(max(p["u_eta"], 1e-12), 1 - 1e-12)))
        for tag, u in (("etaind", u_ind),
                       ("etarho", ndtr(np.sqrt(RHO_ETA) * z_eta + np.sqrt(1 - RHO_ETA) * eps))):
            ec = M.tri_ppf(u, lo, md, hi)
            e0 = np.minimum(eta0_of(p, G, ec), ec)
            Rx, Rlx = ec[:, None] * Gh, ec[:, None] * (Gh - Gbh)
            Rdx, Rldx = Rx.mean(axis=1), Rlx.mean(axis=1)
            Ccfx, Ocfx = cf_cost(p, Rdx, ec, e0)
            Ax = M.route_A(Rx, Rdx, np.minimum(Rx, Rdx[:, None]), p, "scale", False, Ccfx, Ocfx)
            Ax["lcoh"] = np.where(Rdx > 0, Ax["lcoh"], np.nan)
            Alx = M.route_A(Rlx, Rldx, np.minimum(Rlx, Rldx[:, None]), p, "scale", False, Ccfx, Ocfx)
            Bx = M.route_B(Rdx, np.minimum(Rx, Rdx[:, None]), p, Ccfx, Ocfx)
            metrics(tag, Ax, Alx, Bx, i, n)
        # ---------------- composition error common to each department
        Gx, Gbx = generation(p, z_comp=z_dept[dept_idx])
        e0 = eta0_of(p, Gx, eta_c)
        Rx, Rlx = eta_c[:, None] * Gx[:, M.IH:], eta_c[:, None] * (Gx[:, M.IH:] - Gbx[:, M.IH:])
        Rdx, Rldx = Rx.mean(axis=1), Rlx.mean(axis=1)
        Ccfx, Ocfx = cf_cost(p, Rdx, eta_c, e0)
        Ax = M.route_A(Rx, Rdx, np.minimum(Rx, Rdx[:, None]), p, "scale", False, Ccfx, Ocfx)
        Ax["lcoh"] = np.where(Rdx > 0, Ax["lcoh"], np.nan)
        Alx = M.route_A(Rlx, Rldx, np.minimum(Rlx, Rldx[:, None]), p, "scale", False, Ccfx, Ocfx)
        Bx = M.route_B(Rdx, np.minimum(Rx, Rdx[:, None]), p, Ccfx, Ocfx)
        metrics("compdept", Ax, Alx, Bx, i, n)
        # ---------------- reformer CAPEX from the predictive distribution, common to all sites
        sp = float(np.sqrt(M.capex["sigma"] ** 2 + M.capex["lnC1500_se_fit"] ** 2))
        q = dict(p, cREF=M.capex["C1500"] * float(np.exp(sp * p["z_cREF"])))
        Ax = M.route_A(R, Rd, U, q, "scale", False, Ccf, Ocf)
        Ax["lcoh"] = np.where(ok, Ax["lcoh"], np.nan)
        metrics("capexpred", Ax, None, None, i, n)
        # ---------------- reformer CAPEX from the mixed model with a random intercept by study
        q = dict(p, cREF=MIX["C1500"] * float(np.exp(MIX["sigma_total"] * p["z_cREF"])),
                 nR=MIX["n"] + (p["nR"] - M.capex["n"]) * MIX["n_se"] / M.capex["n_se"])
        Ax = M.route_A(R, Rd, U, q, "scale", False, Ccf, Ocf)
        Ax["lcoh"] = np.where(ok, Ax["lcoh"], np.nan)
        metrics("capexmix", Ax, None, None, i, n)
        # ---------------- collection by cover stage
        Hb_, groups, stg = dyn_matrices(p["u_eta"])
        _, _, Rdy, Rly = generation(p, dyn=(Hb_, groups))
        # sites that are not sanitary landfills keep their constant efficiency
        Rdy = np.where(san[:, None], Rdy, eta_c[:, None] * G)[:, M.IH:]
        Rly = np.where(san[:, None], Rly, eta_c[:, None] * (G - Gb))[:, M.IH:]
        Rdd, Rldd = Rdy.mean(axis=1), Rly.mean(axis=1)
        eta_eff = np.where(Gh.mean(axis=1) > 0, Rdd / np.maximum(Gh.mean(axis=1), 1e-9), 0.0)
        Ccfx, Ocfx = cf_cost(p, Rdd, eta_eff, np.minimum(eta0, eta_eff))
        Ay = M.route_A(Rdy, Rdd, np.minimum(Rdy, Rdd[:, None]), p, "scale", False, Ccfx, Ocfx)
        Ay["lcoh"] = np.where(Rdd > 0, Ay["lcoh"], np.nan)
        Aly = M.route_A(Rly, Rldd, np.minimum(Rly, Rldd[:, None]), p, "scale", False, Ccfx, Ocfx)
        By = M.route_B(Rdd, np.minimum(Rdy, Rdd[:, None]), p, Ccfx, Ocfx)
        metrics("dyn", Ay, Aly, By, i, n)
        n["dyn_eta_daily"], n["dyn_eta_inter"], n["dyn_eta_final"] = stg
        n["dyn_eta_eff_w"] = float(Rdy[act].sum() / max(Gh[act].sum(), 1e-9))
        n["dyn_eta_eff_lic_w"] = float(Rly[act].sum() / max((Gh - Gbh)[act].sum(), 1e-9))
        n["base_eta_w"] = float(R[act].sum() / max(Gh[act].sum(), 1e-9))
        yr["dyn"] = Ay["H2_y"][act].sum(axis=0)
        yr["dyn_lic"] = Aly["H2_y"][act].sum(axis=0)
        yr["rec_dyn"] = Rdy[act].sum(axis=0)
        yr["rec_dyn_lic"] = Rly[act].sum(axis=0)
        # ---------------- recovered methane priced at its netback as electricity
        nb = netback(p, Rd, U)
        Ao = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf, {"gas_price": nb})
        Ao["lcoh"] = np.where(ok, Ao["lcoh"], np.nan)
        Bo = M.route_B(Rd, U, p, Ccf, Ocf, {"gas_price": nb})
        metrics("opp", Ao, None, Bo, i, n)
        Hs = A["H2_y"].mean(axis=1)[act]
        n["opp_netback_w"] = float(np.sum(nb[act] * U[act].mean(axis=1)) / max(U[act].mean(axis=1).sum(), 1e-9))
        n["opp_share_pos"] = float(Hs[nb[act] > 0].sum() / max(Hs.sum(), 1e-9))
        # ---------------- capture fraction, CO2 transport and storage, storage losses
        for tag, o in (("cc85", {"capt": CAPT_LO}), ("cc95", {"capt": CAPT_HI}),
                       ("ccnots", {"co2_ts": 0.0}), ("ccleak1", {"co2_leak": LEAK_LO}),
                       ("ccleak10", {"co2_leak": LEAK_HI})):
            Ax = M.route_A(R, Rd, U, p, "scale", True, Ccf, Ocf, o)
            Ax["lcoh"] = np.where(ok, Ax["lcoh"], np.nan)
            metrics(tag, None, None, None, i, n, Ax)
        n["co2_ts"] = p["co2_ts"]
        # ---------------- cleanup and upgrading under a heavier contaminant load
        for tag, o in (("orng15", {"orng_mult": 1.5}), ("orng20", {"orng_mult": 2.0}),
                       ("eaux15", {"eaux_mult": 1.5})):
            Ax = M.route_A(R, Rd, U, p, "scale", False, Ccf, Ocf, o)
            Ax["lcoh"] = np.where(ok, Ax["lcoh"], np.nan)
            metrics(tag, Ax, None, None, i, n)
        # ---------------- Route B with the generator fixed at the modal technology
        Bt = M.route_B(Rd, U, p, Ccf, Ocf, {"tech": TECHB_MODE})
        metrics("techfix", None, None, Bt, i, n)
        n["i"] = i
        rows.append(n)
        years.append({k: v for k, v in yr.items()})
        if (d + 1) % 250 == 0:
            print(f"  realization {d + 1}/{NDRAW}  {time.time() - t0:.0f} s", flush=True)
    nat = pd.DataFrame(rows)
    nat.to_csv(OUT / "variants_national.csv", index=False, float_format="%.6g")
    ykeys = list(years[0])
    arr = {k: np.array([y[k] for y in years]) for k in ykeys}
    np.savez_compressed(OUT / "variants_years.npz", years=M.YEARS, **arr)
    q = nat.quantile([0.1, 0.5, 0.9]).T
    pd.set_option("display.max_rows", 400)
    print(q.round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
