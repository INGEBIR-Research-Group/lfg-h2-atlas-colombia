"""
Atlas of green hydrogen from landfill gas in Colombia, stages 2-6 of the chain.

Category-resolved first-order decay at every canonical disposal site, capture by site class, energy-recovery sizing with the LFGcost-Web technology set, conversion to hydrogen by catalytic reforming (Route A) and by generator plus alkaline electrolysis (Route B), discounted-cash-flow levelized cost over 2026-2040, cradle-to-gate carbon intensity with methane slip, and Monte Carlo propagation.

Inputs (all produced by the scripts named)
    results/site_catalog.csv                  scripts/02_site_catalog.py
    results/waste_municipality_site_year.csv  scripts/03_build_waste.py
    results/waste_site_year.csv               scripts/03_build_waste.py
    results/site_precipitation.csv            scripts/05_precip_zones.py
    results/capex_final.json                  scripts/08_capex_regression.py
    results/composition_municipality.csv   scripts/04_build_composition.py
    inputs/site_facts.csv                     curated baseline capture evidence

Outputs
    results/site_results.csv     one row per site, medians and 10th-90th percentiles
    results/mc_national.csv      national aggregates over the sites active in 2024, one row per draw
    results/mc_params.csv        sampled global parameters, one row per draw
    results/mc_site_draws.npz    per-site draws of the headline quantities (float32)
    results/model_meta.json      settings of the run

Run from the project root:
    python scripts/11_lfg_h2_model.py [NDRAW]
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, INP = ROOT / "results", ROOT / "inputs"
RES.mkdir(exist_ok=True)

SEED = 20260910
NDRAW = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10000)
# --split draws the reformer capital as a common level (standard error of the fitted mean) times an independent residual per site, instead of one common residual; it writes only results/mc_national_split.csv and leaves every other output untouched
SPLIT = "--split" in sys.argv[1:]
RNG = np.random.default_rng(SEED)
RNG_SITE = np.random.default_rng(SEED + 1)
# the burner slip and the parameters of the PSA recovery and of CO2 transport draw from two further streams of the same seed, so that each group keeps its draws when another changes
RNG_BURN = np.random.default_rng(SEED + 2)
RNG_PSA = np.random.default_rng(SEED + 3)

Y0, Y1 = 2026, 2040                        # project evaluation window
YEARS = np.arange(Y0, Y1 + 1)
NY = len(YEARS)
EV0 = 2019                                 # first evaluated year (baseline evidence)
EVAL = np.arange(EV0, Y1 + 1)
IH = int(np.searchsorted(EVAL, Y0))
DEP0 = 1985
DEP = np.arange(DEP0, Y1 + 1)

# ---------------------------------------------------------------- constants
RHO_CH4 = 0.678            # kg/m3 at 15 C and 1 atm
HHV_CH4 = 37.7             # MJ/m3
LHV_CH4 = 33.93            # MJ/m3, 50.05 MJ/kg
LHV_H2 = 119.96            # MJ/kg
KG_H2_PER_M3 = LHV_CH4 / LHV_H2           # kg H2 per m3 CH4 at unit LHV efficiency
R_CO2 = 44.009 / 16.043
H = 8760.0
AV = 0.93                  # availability of reformer, electrolyzer and generators
GWP = 27.0                 # AR6 GWP100, methane of non-fossil origin
GWP_ALT = {"AR5": 28.0, "AR5_feedback": 34.0, "AR6_fossil": 29.8}
OX = 0.10                  # soil-cover oxidation at a sanitary landfill
CAPT = 0.90                # capture fraction of the process CO2 stream
BOILER = 0.85              # boiler efficiency for solvent regeneration heat
DEPTH, RHO_W = 25.0, 0.9   # m, Mg/m3 compacted
WIP_YEAR = 2032            # waste in place that sets the well field, mid-horizon
STACK_H = 75000.0          # h
STACK_SHARE = 0.35
MTUR_LIFE = 10

CEPCI = {2006: 499.6, 2007: 525.4, 2008: 575.4, 2013: 567.3,
         2018: 603.1, 2019: 607.5, 2020: 596.2, 2023: 797.9}
ESC = {y: CEPCI[2023] / v for y, v in CEPCI.items()}

# Colombia Landfill Gas Model, k by decay category and precipitation zone (wettest first)
K = np.array([[0.400, 0.340, 0.260, 0.180, 0.100],
              [0.170, 0.150, 0.120, 0.090, 0.050],
              [0.070, 0.060, 0.048, 0.036, 0.020],
              [0.035, 0.030, 0.024, 0.018, 0.010]])
L0C = np.array([70.0, 113.0, 186.0, 200.0])

# class: (MCF, eta_c lo, mode, hi). The sanitary-landfill field follows the cover-stage defaults of in-situ campaigns (Giordano et al. 2024).
SITE_CLASS = {
    "Sanitary Landfill": (1.0, 0.41, 0.60, 0.71),
    "Open Dump":         (0.8, 0.00, 0.05, 0.20),
    "Transitional Cell": (0.8, 0.20, 0.35, 0.50),
    "Integral Plant":    (0.8, 0.20, 0.35, 0.50),
    "Contingency Cell":  (0.8, 0.15, 0.30, 0.45),
    "Burial":            (0.4, 0.00, 0.00, 0.05),
    "Water Body":        (0.0, 0.00, 0.00, 0.00),
    "Treatment Plant":   (0.0, 0.00, 0.00, 0.00),
    "Open Burning":      (0.0, 0.00, 0.00, 0.00),
}
# LFGcost-Web generators: heat rate MJ/kWh (HHV), net/gross, capacity factor, P range kW
GEN = {
    "ENG":  (11.9, 0.93, 0.93, 800.0, 1e9),
    "SENG": (19.2, 0.92, 0.93, 100.0, 1000.0),
    "TUR":  (13.7, 0.88, 0.93, 3000.0, 1e9),
    "MTUR": (14.8, 0.83, 0.93, 30.0, 750.0),
}
GEN_NAMES = list(GEN)
GEN_BASE = {"ENG": 2013, "SENG": 2008, "TUR": 2008, "MTUR": 2006}

# reformer overall efficiency, LHV of hydrogen over LHV of feed plus fuel
E_SMALL = (0.52, 0.60, 0.72)    # at 1.5 t/d
E_LARGE = (0.72, 0.76, 0.77)    # at 216 t/d
Q_SMALL = 1500.0 * LHV_H2 / (0.60 * LHV_CH4) / 24.0      # m3 CH4/h at 1.5 t/d
Q_LARGE = 216000.0 * LHV_H2 / (0.76 * LHV_CH4) / 24.0    # m3 CH4/h at 216 t/d
E_SINGH = 0.35                  # implied by the fuel ratio of the process simulation
# hydrogen recovery of the pressure-swing adsorption unit at the two anchors of E. The reformer and shift reactors form a share eta_ref of the stoichiometric hydrogen of the feed, and the adsorption unit delivers a share R_PSA of it; the rest leaves in the tail gas and is burned in the furnace. Interpolated in the logarithm of the flow, as E.
R_PSA_SMALL = (0.70, 0.78, 0.86)
R_PSA_LARGE = (0.86, 0.89, 0.92)
# emissions of CO2 transport and storage in the capture variant, kg CO2e per kg CO2 stored: pipeline (about 0.01 to 0.03) to road tankers (about 0.03 to 0.09) over 500 to 1,000 km to a storage site, injection included
CO2_TS = (0.01, 0.03, 0.09)
RNG_Q_LO, RNG_Q_HI = 1700.0, 10200.0      # LFGcost range of the upgrading correlation
# decay timing of the Colombia Landfill Gas Model: each year's deposit is split into ten equal parts placed at the middle of each tenth of the year, generation starts six months after placement, and the annual rate is read at the middle of the calendar year
LAG = 0.5                                   # yr
NSEC = 10
SEC_OFF = (np.arange(1, NSEC + 1) - 0.5) / NSEC
# methane slip of the reformer furnace and of the capture boiler, fraction of the methane they burn: 1 to 5 kg CH4 per TJ, the inventory defaults for landfill gas burned in energy industries and in commercial boilers (IPCC 2006, Vol. 2, Tables 2.2 and 2.4)
S_BURN = (0.00005, 0.00005, 0.00025)
# organic-diversion variant: share of food and garden waste diverted from deposits, reached linearly between the first diverted year and DIV_FULL and held afterwards
DIV_LEVEL, DIV_Y0, DIV_FULL = 0.50, 2026, 2035
# compression of electrolytic hydrogen from atmospheric pressure to the 30 bar of the reformer, kWh/kg: isothermal work of 1.16 kWh/kg at a compressor efficiency of 0.6
E_COMP = 1.94


def tri_ppf(u, lo, mode, hi):
    """Inverse cdf of the triangular distribution, vectorised."""
    u, lo, mode, hi = (np.array(a, dtype=float) for a in np.broadcast_arrays(
        np.asarray(u, float), np.asarray(lo, float), np.asarray(mode, float),
        np.asarray(hi, float)))
    w = hi - lo
    out = np.array(mode, dtype=float, copy=True)
    ok = w > 0
    fc = np.where(ok, (mode - lo) / np.where(ok, w, 1.0), 0.5)
    left = ok & (u < fc)
    right = ok & ~left
    out[left] = lo[left] + np.sqrt(u[left] * w[left] * (mode[left] - lo[left]))
    out[right] = hi[right] - np.sqrt((1 - u[right]) * w[right] * (hi[right] - mode[right]))
    return out


def lognorm_5_95(u, lo, hi):
    """Lognormal whose 5th and 95th percentiles are lo and hi, from a normal quantile."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(float(u))
    mu, sg = 0.5 * (np.log(lo) + np.log(hi)), (np.log(hi) - np.log(lo)) / (2 * 1.6449)
    return float(np.exp(mu + sg * z))


def decay_kernel(k, lag=None):
    """Fraction of k L0 M generated in each evaluated year by the deposit of each year.

    Rows are deposit years, columns evaluated years. Each deposit is split into NSEC equal parts placed at the middle of their tenth of the year, a part starts to generate LAG years after placement, and the first-order rate is read at the middle of the calendar year. With LAG = 0.5 a deposit starts to generate in the following year, and the cumulative generation of a deposit equals L0 M to within 0.01 percent."""
    lag = LAG if lag is None else lag
    tau = AGE[:, :, None] + 0.5 - SEC_OFF[None, None, :] - lag
    return np.where(tau >= 0.0, np.exp(-k * np.clip(tau, 0.0, None)), 0.0).mean(axis=2)


# ---------------------------------------------------------------- inputs
cat = pd.read_csv(RES / "site_catalog.csv", index_col=0)
zones = pd.read_csv(RES / "site_precipitation.csv").set_index("site_id")
comp = pd.read_csv(RES / "composition_municipality.csv", dtype={"municipality_code": str})
comp["municipality_code"] = comp["municipality_code"].str.zfill(5)
wm = pd.read_csv(RES / "waste_municipality_site_year.csv", dtype={"served_dane_code": str})
wm["served_dane_code"] = wm["served_dane_code"].str.zfill(5)
ws = pd.read_csv(RES / "waste_site_year.csv")
capex = json.loads((RES / "capex_final.json").read_text())
facts = pd.read_csv(INP / "site_facts.csv").set_index("site_id")

SITES = list(cat.index)
NS = len(SITES)
sidx = {s: i for i, s in enumerate(SITES)}
print(f"sites carried: {NS}")

# deposits by decay category, sites x categories x deposit years
f = comp.set_index("municipality_code")[["food", "garden", "paper", "wood", "textiles",
                                          "diapers", "nondeg"]] / 100.0
cats = pd.DataFrame({"c1": f.food + 0.20 * f.diapers, "c2": f.garden,
                     "c3": f.paper + f.textiles + 0.80 * f.diapers, "c4": f.wood})
tier = comp.set_index("municipality_code")["tier"]
wm = wm[wm["year"].between(DEP0, Y1)]
missing = set(wm["served_dane_code"]) - set(cats.index)
if missing:
    raise SystemExit(f"municipalities without composition: {sorted(missing)[:10]}")
Wc = np.zeros((NS, 4, len(DEP)))
si = wm["site_id"].map(sidx).to_numpy()
di = (wm["year"] - DEP0).to_numpy()
cm = cats.loc[wm["served_dane_code"]].to_numpy()
for c in range(4):
    np.add.at(Wc[:, c, :], (si, di), wm["tons"].to_numpy() * cm[:, c])
W = np.zeros((NS, len(DEP)))
np.add.at(W, (si, di), wm["tons"].to_numpy())
meas = wm["served_dane_code"].map(tier).eq("measured").to_numpy()
share_measured = np.zeros(NS)
np.add.at(share_measured, si, np.where(meas, wm["tons"].to_numpy(), 0.0))
share_measured = share_measured / np.maximum(W.sum(1), 1e-9)

# the part of each deposit year beyond the licensed end of life
ws = ws[ws["year"].between(DEP0, Y1)]
beyond = np.zeros((NS, len(DEP)))
np.add.at(beyond, (ws["site_id"].map(sidx).to_numpy(), (ws["year"] - DEP0).to_numpy()),
          (ws["tons_beyond_licence"] / ws["tons"].where(ws["tons"] > 0, np.nan)).fillna(0.0).to_numpy())
Wc_bey = Wc * beyond[:, None, :]
# retained share of food and garden waste in the organic-diversion variant, by deposit year
DIVF = 1.0 - DIV_LEVEL * np.clip((DEP - DIV_Y0 + 1) / (DIV_FULL - DIV_Y0 + 1), 0.0, 1.0)

# precipitation zone, index 0 = wettest
zone = zones["zone_9120"].reindex(SITES).to_numpy()
if np.isnan(zone).any():
    raise SystemExit("sites without a precipitation zone")
zone = zone.astype(int) - 1

# class parameters; a site with no recorded class is drawn by national mass shares
cls = cat["site_type"].to_numpy(dtype=object)
unrec = pd.isna(cls)
mass_share = cat.groupby("site_type")["tons_2010_2024"].sum()
mass_share = mass_share / mass_share.sum()
cls_fill = np.where(unrec, "Sanitary Landfill", cls)
mcf0 = np.array([SITE_CLASS[c][0] for c in cls_fill])
eta_lo = np.array([SITE_CLASS[c][1] for c in cls_fill])
eta_md = np.array([SITE_CLASS[c][2] for c in cls_fill])
eta_hi = np.array([SITE_CLASS[c][3] for c in cls_fill])
sanitary = cls_fill == "Sanitary Landfill"

# baseline capture evidence
cap_t = np.full((NS, 3), np.nan)
cap_f = np.full((NS, 3), np.nan)
cap_years = {}
for s, r in facts.iterrows():
    if s not in sidx or pd.isna(r.get("capture_basis")) or r["capture_basis"] == "none":
        continue
    i = sidx[s]
    if pd.notna(r["capture_ch4_t_mode"]):
        cap_t[i] = [r["capture_ch4_t_lo"], r["capture_ch4_t_mode"], r["capture_ch4_t_hi"]]
    if pd.notna(r["capture_frac_mode"]):
        cap_f[i] = [r["capture_frac_lo"], r["capture_frac_mode"], r["capture_frac_hi"]]
    yrs = str(r["capture_years"]).split("-")
    cap_years[i] = (int(yrs[0]), int(yrs[-1]))
has_t = np.isfinite(cap_t[:, 1])
has_f = np.isfinite(cap_f[:, 1])
print(f"baseline capture evidence at {int(has_t.sum() + has_f.sum())} sites")

AGE = EVAL[None, :] - DEP[:, None]
CUM = np.cumsum(W, axis=1)
wip = CUM[:, int(np.searchsorted(DEP, WIP_YEAR))]
area_ha = np.clip(wip / RHO_W / DEPTH / 1e4, 0.1, None)
Nw = 2.47 * area_ha
active = (cat["status_2024"] == "active").to_numpy()
DJ = sidx.get("11001|DONA JUANA")

DF_T = np.arange(1, NY + 1)


def dcf_lcoh(C0, O_fix, var_y, H2_y, i, reps=()):
    """Levelized cost by discounted cash flow; capital at t=0, flows at year end."""
    df = (1.0 + i) ** -DF_T
    num = C0 + O_fix * df.sum() + (var_y * df).sum(axis=1)
    for t, cost in reps:
        num = num + cost * (1.0 + i) ** -t
    den = (H2_y * df).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gen_cost(name, P, Egross):
    """LFGcost-Web capital and O&M of a generator, base-year USD."""
    P = np.asarray(P, float)
    if name == "ENG":
        Cg, Og = 1300.0 * P + 1.35e6, 0.025 * Egross
    elif name == "SENG":
        Cg, Og = 2300.0 * P, 0.024 * Egross
    elif name == "TUR":
        Cg = np.where(2340.0 - 0.103 * P < 1015.0, 1015.0 * P, 2340.0 * P - 0.103 * P ** 2) + 2.50e5
        Og = 0.0144 * Egross
    else:
        Cg = 1.9278e4 * np.clip(P, 1.0, None) ** 0.6207
        Og = (0.0736 - 0.0094 * np.log(np.clip(P, 1.001, None))) * Egross
    return Cg, Og


def upgrading_cost(Q, fL, n_up):
    """Cleanup and upgrading train, LFGcost-Web high-Btu correlation, USD 2019.

    Parallel trains above the upper bound of the correlation, and a power-law extension below its lower bound. The pipeline allowance of the correlation is not included, since the gas is used on site."""
    Q = np.clip(np.asarray(Q, float), 1e-6, None)
    N = np.ceil(Q / RNG_Q_HI)
    q = Q / N
    hi = 6.00e6 * np.exp(1.766e-4 * np.clip(q, RNG_Q_LO, None))
    lo = 6.00e6 * np.exp(1.766e-4 * RNG_Q_LO) * (q / RNG_Q_LO) ** n_up
    C = N * np.where(q >= RNG_Q_LO, hi, lo)
    O = N * (147.0 * q + 1.48e5)
    return fL * C * ESC[2019], fL * O * ESC[2019]


def cheapest_generator(P_kW, Egross_y, i, fL):
    """Least-cost LFGcost generator that can supply P_kW, smallest unit at its floor."""
    best = np.full(P_kW.shape, np.inf)
    tech = np.zeros(P_kW.shape, dtype=int)
    Cg_b = np.zeros(P_kW.shape)
    P_b = np.zeros(P_kW.shape)
    crf = i * (1 + i) ** NY / ((1 + i) ** NY - 1)
    for ti, name in enumerate(GEN_NAMES):
        HR, epar, CF, Pmin, Pmax = GEN[name]
        Puse = np.maximum(P_kW, Pmin) if name == "MTUR" else P_kW
        feas = (Puse >= Pmin) & (Puse <= Pmax)
        Cg, Og = gen_cost(name, Puse, Egross_y.mean(axis=1))
        Cg, Og = fL * Cg * ESC[GEN_BASE[name]], fL * Og * ESC[GEN_BASE[name]]
        cost = np.where(feas, crf * Cg + Og, np.inf)
        upd = cost < best
        best = np.where(upd, cost, best)
        tech = np.where(upd, ti, tech)
        Cg_b = np.where(upd, Cg, Cg_b)
        P_b = np.where(upd, Puse, P_b)
    return tech, Cg_b, P_b


def draw_params():
    p = {}
    p["u_k"] = RNG.random(4)
    p["l0m"] = tri_ppf(RNG.random(4), 0.75, 1.0, 1.25)
    p["u_eta"] = RNG.random()
    p["F"] = float(tri_ppf(RNG.random(), 0.45, 0.50, 0.60))
    p["u_E"] = RNG.random()
    p["eta_ref"] = float(tri_ppf(RNG.random(), 0.65, 0.75, 0.88))
    p["s_up"] = float(tri_ppf(RNG.random(), 0.001, 0.005, 0.020))
    p["s_eng"] = float(tri_ppf(RNG.random(), 0.004, 0.016, 0.033))
    p["s_mtur"] = float(tri_ppf(RNG.random(), 0.0001, 0.003, 0.009))
    p["s_tur"] = 0.0002
    p["sec"] = float(tri_ppf(RNG.random(), 50.0, 53.0, 57.0))
    p["nR"] = float(tri_ppf(RNG.random(), *[capex["n_ci_t"][0], capex["n"], capex["n_ci_t"][1]]))
    p["z_cREF"] = float(RNG.standard_normal())
    if SPLIT:
        p["cREF"] = capex["C1500"] * np.exp(capex["lnC1500_se_fit"] * p["z_cREF"]
                                            + capex["sigma"] * RNG_SITE.standard_normal(NS))
    else:
        p["cREF"] = capex["C1500"] * float(np.exp(capex["sigma"] * p["z_cREF"]))
    p["cEL"] = float(tri_ppf(RNG.random(), 0.9e6, 1.8e6, 2.6e6))
    p["nE"] = float(tri_ppf(RNG.random(), 0.70, 0.78, 0.90))
    p["fL"] = lognorm_5_95(RNG.random(), 0.75, 1.25)
    p["i"] = float(tri_ppf(RNG.random(), 0.06, 0.09, 0.12))
    p["ccc"] = float(tri_ppf(RNG.random(), 45.0, 60.0, 75.0)) * ESC[2018]
    p["tns"] = float(tri_ppf(RNG.random(), 20.0, 40.0, 80.0))
    p["q_regen"] = float(tri_ppf(RNG.random(), 2.2, 2.9, 3.1))
    p["e_cc"] = float(tri_ppf(RNG.random(), 0.08, 0.12, 0.18))
    p["pe"] = float(tri_ppf(RNG.random(), 0.055, 0.088, 0.095))
    p["e_aux"] = float(tri_ppf(RNG.random(), 0.15, 0.25, 0.35))
    p["n_up"] = float(tri_ppf(RNG.random(), 0.50, 0.65, 0.80))
    p["u_cap"] = RNG.random(NS)
    p["u_unrec"] = RNG.random(NS)
    p["z_comp"] = RNG.standard_normal((NS, 4))
    p["s_burn"] = float(tri_ppf(RNG_BURN.random(), *S_BURN))
    p["u_psa"] = RNG_PSA.random()
    p["co2_ts"] = float(tri_ppf(RNG_PSA.random(), *CO2_TS))
    return p


def scale_fraction(Rd):
    """Position of the design flow between the two anchors, 0 to 1 in the logarithm of flow."""
    Qd = Rd / (H * AV)
    return np.clip(np.log(np.clip(Qd, 1e-9, None) / Q_SMALL) / np.log(Q_LARGE / Q_SMALL), 0.0, 1.0)


def psa_recovery(Rd, p):
    """Hydrogen recovery of the adsorption unit at the design flow Rd."""
    x = scale_fraction(Rd)
    rs = tri_ppf(p["u_psa"], *R_PSA_SMALL)
    rl = tri_ppf(p["u_psa"], *R_PSA_LARGE)
    return rs + (rl - rs) * x


def route_A(R, Rd, U, p, E_mode="scale", ccs=False, Ccf=0.0, Ocf=0.0, opt=None):
    """Reformer island fed from the recovered stream, own electricity from an engine.

    opt holds the switches of the variants, all off by default:
    fmin       minimum load of a train as a share of its design flow; a year whose gas falls below it produces nothing and flares its gas, while the fixed costs continue
    trains     number of identical parallel trains of reformer and upgrading unit, each sized at Rd / trains, so that the plant can turn down to fmin / trains
    gas_price  price of the recovered methane drawn by the plant, USD per m3, one per site
    orng_mult  multiplier of the operating cost of the cleanup and upgrading train
    eaux_mult  multiplier of the auxiliary electricity of that train
    capt       capture fraction of the process CO2 (default CAPT)
    co2_ts     emission of CO2 transport and storage, kg CO2e per kg CO2 stored (default p["co2_ts"])
    co2_leak   share of the stored CO2 that escapes from storage (default 0)
    """
    o = {} if opt is None else opt
    fmin, trains = o.get("fmin", 0.0), int(o.get("trains", 1))
    capt = o.get("capt", CAPT)
    i = p["i"]
    if E_mode == "scale":
        el = tri_ppf(p["u_E"], *E_SMALL)
        eh = tri_ppf(p["u_E"], *E_LARGE)
        E = el + (eh - el) * scale_fraction(Rd)
    else:
        E = np.full(Rd.shape, E_SINGH)
    rpsa = psa_recovery(Rd, p)
    phi = np.clip(1.2051 * rpsa * p["eta_ref"] / E - 1.0, 0.0, None)
    ff = 1.0 / (1.0 + phi)                                 # feed share of island methane
    c_proc = RHO_CH4 * R_CO2 * p["eta_ref"] * ff           # kg process CO2 per m3 island CH4
    a_up = p["s_up"] / (1.0 - p["s_up"]) * ff
    e_aux = p["e_aux"] * o.get("eaux_mult", 1.0)
    e_per = e_aux * ff / p["F"] + (p["e_cc"] * capt * c_proc if ccs else 0.0)
    a_boil = (p["q_regen"] * capt * c_proc / (BOILER * LHV_CH4)) if ccs else 0.0
    # first pass with a large engine to size the auxiliary load, then the chosen unit
    HR0, ep0, CF0 = GEN["ENG"][0], GEN["ENG"][1], GEN["ENG"][2]
    den0 = 1.0 + a_up + e_per * HR0 / (ep0 * HHV_CH4) + a_boil
    Paux = e_per * (Rd / den0) / (H * CF0)
    Egross0 = (e_per / ep0 / den0)[:, None] * U
    tech, Cg, Paux_used = cheapest_generator(Paux, Egross0, i, p["fL"])
    HR = np.array([GEN[n][0] for n in GEN_NAMES])[tech]
    ep = np.array([GEN[n][1] for n in GEN_NAMES])[tech]
    slip = np.array([p["s_eng"], p["s_eng"], p["s_tur"], p["s_mtur"]])[tech]
    a_eng = e_per * HR / (ep * HHV_CH4)
    a = a_up + a_eng + a_boil
    # years in which the gas cannot keep one train above its minimum load
    on = (U >= fmin * Rd[:, None] / trains) if fmin > 0 else np.ones(U.shape, dtype=bool)
    xd = Rd / (1.0 + a)
    xy = U / (1.0 + a)[:, None] * on
    H2_y = (E * KG_H2_PER_M3)[:, None] * xy
    H2_d = E * KG_H2_PER_M3 * xd
    # capital and operating cost
    Qfeed = xd * ff / (1.0 - p["s_up"]) / p["F"] / (H * AV)
    Crng, Orng = upgrading_cost(Qfeed / trains, p["fL"], p["n_up"])
    Crng, Orng = trains * Crng, trains * Orng * o.get("orng_mult", 1.0)
    S = H2_d / (365.0 * AV)
    Cref = trains * p["cREF"] * (np.clip(S / trains, 1e-9, None) / 1500.0) ** p["nR"]
    Oref = 0.04 * Cref
    Egr_y = (e_per / ep)[:, None] * xy
    Og_y = np.zeros_like(xy)
    for ti, name in enumerate(GEN_NAMES):
        sel = tech == ti
        if sel.any():
            _, og = gen_cost(name, np.broadcast_to(Paux_used[sel][:, None], Egr_y[sel].shape),
                             Egr_y[sel])
            Og_y[sel] = p["fL"] * og * ESC[GEN_BASE[name]]
    C0 = Ccf + Crng + Cref + Cg
    O_fix = Ocf + Orng + Oref
    var = Og_y.copy()
    if o.get("gas_price") is not None:
        var = var + np.asarray(o["gas_price"])[:, None] * U * on
    co2_y = (c_proc[:, None] * xy) * capt                   # kg captured per year
    if ccs:
        var = var + (p["ccc"] + p["tns"]) * co2_y / 1000.0
    reps = [(MTUR_LIFE, np.where(tech == 3, Cg, 0.0))]
    lcoh = dcf_lcoh(C0, O_fix, var, H2_y, i, reps)
    # carbon intensity, kg CO2e per kg H2
    # methane that escapes the furnace (fuel and the unconverted feed methane of the tail gas, an upper bound) and the capture boiler, per unit of island methane
    burned = 1.0 - ff * p["eta_ref"] + a_boil
    ch4_emit = a_up + a_eng * slip + p["s_burn"] * burned
    ci = RHO_CH4 * GWP * ch4_emit / (E * KG_H2_PER_M3)
    if ccs:
        ts = o.get("co2_ts", p.get("co2_ts", 0.0)) + o.get("co2_leak", 0.0)
        ci = ci - capt * c_proc / (E * KG_H2_PER_M3) * (1.0 - ts)
    return dict(H2_y=H2_y, H2_d=H2_d, lcoh=lcoh, ci=ci, E=E, tech=tech, a=a,
                ch4_emit=ch4_emit, co2=co2_y.mean(axis=1), S=S, C0=C0, Qfeed=Qfeed,
                rpsa=rpsa, phi=phi, on=on, var=var, O_fix=O_fix)


def route_B(Rd, U, p, Ccf, Ocf, opt=None):
    """Generator plus alkaline electrolyzer, least cost among the feasible machines.

    opt holds the switches of the variants, all off by default:
    fmin       minimum load of the generator as a share of its design flow; a year whose gas falls below it produces nothing and flares its gas
    gas_price  price of the recovered methane burned by the generator, USD per m3, per site
    tech       generator index forced at each site (-1 leaves the least-cost choice)
    """
    o = {} if opt is None else opt
    fmin = o.get("fmin", 0.0)
    forced = o.get("tech")
    i = p["i"]
    best = np.full(Rd.shape, np.inf)
    out = dict(H2_y=np.zeros_like(U), tech=np.full(Rd.shape, -1), Pel=np.zeros_like(Rd),
               E=np.zeros_like(Rd), ci=np.full(Rd.shape, np.nan), lcoh=np.full(Rd.shape, np.nan),
               C0=np.zeros_like(Rd))
    thermal_d = Rd * HHV_CH4
    slip_of = {"ENG": p["s_eng"], "SENG": p["s_eng"], "TUR": p["s_tur"], "MTUR": p["s_mtur"]}
    on = (U >= fmin * Rd[:, None]) if fmin > 0 else np.ones(U.shape, dtype=bool)
    for ti, name in enumerate(GEN_NAMES):
        HR, epar, CF, Pmin, Pmax = GEN[name]
        P = thermal_d / (HR * H * CF)
        feas = (P >= Pmin) & (P <= Pmax)
        if forced is not None:
            feas = feas & ((np.asarray(forced) == ti) | (np.asarray(forced) < 0))
        if not feas.any():
            continue
        Egross_y = U * HHV_CH4 / HR * on
        Enet_y = Egross_y * epar
        Cg, _ = gen_cost(name, P, Egross_y.mean(axis=1))
        _, Og_y = gen_cost(name, np.broadcast_to(P[:, None], U.shape), Egross_y)
        Cg = p["fL"] * Cg * ESC[GEN_BASE[name]]
        Og_y = p["fL"] * Og_y * ESC[GEN_BASE[name]]
        Pel = (U * HHV_CH4 / HR * epar).mean(axis=1) / (H * AV)
        Cel = p["cEL"] * (np.clip(Pel, 1e-6, None) / 1000.0) ** p["nE"]
        Oel = 0.02 * Cel
        H2_y = Enet_y / p["sec"]
        var = Og_y + 12e-3 * H2_y * 1.50
        if o.get("gas_price") is not None:
            var = var + np.asarray(o["gas_price"])[:, None] * U * on
        t_stack = STACK_H / (H * AV)
        reps = [(t_stack, STACK_SHARE * Cel)]
        if name == "MTUR":
            reps.append((MTUR_LIFE, Cg))
        C0 = Ccf + Cg + Cel
        lc = dcf_lcoh(C0, Ocf + Oel, var, H2_y, i, reps)
        upd = feas & np.isfinite(lc) & (lc < best)
        best = np.where(upd, lc, best)
        out["H2_y"] = np.where(upd[:, None], H2_y, out["H2_y"])
        out["tech"] = np.where(upd, ti, out["tech"])
        out["Pel"] = np.where(upd, Pel, out["Pel"])
        out["E"] = np.where(upd, Enet_y.mean(axis=1), out["E"])
        out["C0"] = np.where(upd, C0, out["C0"])
        ci = RHO_CH4 * GWP * slip_of[name] * HR * p["sec"] / (HHV_CH4 * epar)
        out["ci"] = np.where(upd, ci, out["ci"])
    out["lcoh"] = np.where(np.isfinite(best), best, np.nan)
    return out


def one_draw(p):
    # decay constants and yields
    kz = np.empty((4, 5))
    for c in range(4):
        for z in range(5):
            kz[c, z] = tri_ppf(p["u_k"][c], K[c, min(z + 1, 4)], K[c, z], K[c, max(z - 1, 0)])
    l0 = L0C * p["l0m"]
    # composition perturbation on the simplex, per site, tighter where measured
    tot = np.maximum(W.sum(axis=1), 1e-9)
    wbar = np.clip(Wc.sum(axis=2) / tot[:, None], 1e-6, None)
    rest = np.clip(1.0 - wbar.sum(axis=1, keepdims=True), 1e-6, None)
    sd = np.where(share_measured[:, None] > 0.5, 0.10, 0.20)
    alr = np.log(wbar / rest) + sd * p["z_comp"]
    e = np.exp(alr)
    wnew = e / (1.0 + e.sum(axis=1, keepdims=True))
    rmul = wnew / wbar
    # class of unrecorded sites and methane correction factor
    mcf = mcf0.copy()
    lo, md, hi = eta_lo.copy(), eta_md.copy(), eta_hi.copy()
    if unrec.any():
        names = mass_share.index.to_numpy()
        cum = np.cumsum(mass_share.to_numpy())
        for j in np.flatnonzero(unrec):
            c = names[min(np.searchsorted(cum, p["u_unrec"][j]), len(names) - 1)]
            mcf[j], lo[j], md[j], hi[j] = SITE_CLASS[c]
    eta_c = tri_ppf(p["u_eta"], lo, md, hi)             # common quantile across sites
    G = np.zeros((NS, len(EVAL)))
    Gb = np.zeros((NS, len(EVAL)))
    Gv = np.zeros((NS, len(EVAL)))        # organic-diversion variant
    Gbv = np.zeros((NS, len(EVAL)))
    for z in range(5):
        sel = zone == z
        if not sel.any():
            continue
        for c in range(4):
            D = decay_kernel(kz[c, z])
            coef = kz[c, z] * l0[c] * rmul[sel, c]
            G[sel] += coef[:, None] * (Wc[sel, c, :] @ D)
            Gb[sel] += coef[:, None] * (Wc_bey[sel, c, :] @ D)
            dv = DIVF if c < 2 else 1.0
            Gv[sel] += coef[:, None] * ((Wc[sel, c, :] * dv) @ D)
            Gbv[sel] += coef[:, None] * ((Wc_bey[sel, c, :] * dv) @ D)
    G *= mcf[:, None]
    Gb *= mcf[:, None]
    Gv *= mcf[:, None]
    Gbv *= mcf[:, None]
    # baseline capture already in place
    eta0 = np.zeros(NS)
    for j in np.flatnonzero(has_t | has_f):
        if has_t[j]:
            y0, y1 = cap_years[j]
            gy = G[j, (EVAL >= y0) & (EVAL <= y1)].mean()
            t = tri_ppf(p["u_cap"][j], *cap_t[j])
            eta0[j] = t * 1000.0 / RHO_CH4 / max(gy, 1e-9)
        else:
            eta0[j] = tri_ppf(p["u_cap"][j], *cap_f[j])
    eta0 = np.minimum(eta0, eta_c)
    Gh = G[:, IH:]
    R = eta_c[:, None] * Gh                               # recoverable CH4, m3/yr
    Rl = eta_c[:, None] * (Gh - Gb[:, IH:])               # licensed-only variant
    Rd = R.mean(axis=1)
    U = np.minimum(R, Rd[:, None])
    Rld = Rl.mean(axis=1)
    Ul = np.minimum(Rl, Rld[:, None])
    # collection and flaring, charged at the increment over the existing field
    Qlfg = Rd / p["F"] / H
    incr = np.where(eta_c > 0, 1.0 - eta0 / np.where(eta_c > 0, eta_c, 1.0), 1.0)
    Ccf = incr * p["fL"] * (2.00e4 + Nw * (279.0 * (DEPTH - 3.05) + 1.77e4)
                            + 3329.0 * np.clip(Qlfg, 0, None) ** 0.61) * ESC[2013]
    Ocf = p["fL"] * (2600.0 * Nw + 5100.0) * ESC[2013]
    ok = Rd > 0
    Ccf, Ocf = np.where(ok, Ccf, 0.0), np.where(ok, Ocf, 0.0)
    A = route_A(R, Rd, U, p, "scale", False, Ccf, Ocf)
    Acc = route_A(R, Rd, U, p, "scale", True, Ccf, Ocf)
    As = route_A(R, Rd, U, p, "singh", False, Ccf, Ocf)
    Al = route_A(Rl, Rld, Ul, p, "scale", False, Ccf, Ocf)
    B = route_B(Rd, U, p, Ccf, Ocf)
    # incremental variant: at the sites that already destroy part of their gas, the stream already collected stays with the existing system and only the increment feeds the plant
    has0 = eta0 > 0
    Ri = np.where(has0[:, None], (eta_c - eta0)[:, None] * Gh, R)
    Rid = Ri.mean(axis=1)
    Ai = route_A(Ri, Rid, np.minimum(Ri, Rid[:, None]), p, "scale", False, Ccf, Ocf)
    # organic-diversion variant, base case and licensed-only
    Rv = eta_c[:, None] * Gv[:, IH:]
    Rvd = Rv.mean(axis=1)
    Av = route_A(Rv, Rvd, np.minimum(Rv, Rvd[:, None]), p, "scale", False, Ccf, Ocf)
    Rlv = eta_c[:, None] * (Gv[:, IH:] - Gbv[:, IH:])
    Rlvd = Rlv.mean(axis=1)
    Alv = route_A(Rlv, Rlvd, np.minimum(Rlv, Rlvd[:, None]), p, "scale", False, Ccf, Ocf)
    # electrolytic hydrogen compressed from atmospheric pressure to the reformer's 30 bar
    Bc = route_B(Rd, U, dict(p, sec=p["sec"] + E_COMP), Ccf, Ocf)
    H2A = A["H2_y"].mean(axis=1)
    # avoided methane, consequential and reported apart, kg CO2e per kg H2
    ox = np.where(sanitary, OX, 0.0)
    avoided_t = ((eta_c - eta0)[:, None] * Gh * (1 - ox)[:, None] * RHO_CH4).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        avoidA = np.where(H2A > 0, avoided_t * GWP / H2A, np.nan)
    for d in (A, Acc, As, Al, Ai, Av):
        d["lcoh"] = np.where(ok, d["lcoh"], np.nan)
    return dict(
        H2A=H2A, H2A_cc=Acc["H2_y"].mean(axis=1), H2A_singh=As["H2_y"].mean(axis=1),
        H2A_lic=Al["H2_y"].mean(axis=1), H2A_pot=(A["H2_y"] * R / np.where(U > 0, U, 1.0)).mean(axis=1),
        H2B=B["H2_y"].mean(axis=1),
        lcohA=A["lcoh"], lcohA_cc=Acc["lcoh"], lcohA_singh=As["lcoh"], lcohB=B["lcoh"],
        ciA=np.where(ok, A["ci"], np.nan), ciA_cc=np.where(ok, Acc["ci"], np.nan),
        ciA_cc_slip=np.where(ok, RHO_CH4 * GWP * Acc["ch4_emit"] / (Acc["E"] * KG_H2_PER_M3), np.nan),
        ciB=B["ci"], oppB=np.where(B["H2_y"].mean(axis=1) > 0, p["pe"] * p["sec"], np.nan),
        avoidA=avoidA, avoid_t=avoided_t * GWP / 1000.0,
        gen=Gh.mean(axis=1), rec=R.mean(axis=1), gen_bey=Gb[:, IH:].mean(axis=1) * eta_c,
        eta=eta_c, eta0=eta0, E=A["E"], techA=A["tech"], techB=B["tech"], Pel=B["Pel"],
        co2=A["co2"], co2_cc=Acc["co2"], S=A["S"], Qfeed=A["Qfeed"], capA=A["C0"], capB=B["C0"],
        G_dj=G[DJ, (EVAL >= 2021) & (EVAL <= 2025)].mean() if DJ is not None else np.nan,
        G_2024=G[:, EVAL == 2024][:, 0],
        H2A_incr=Ai["H2_y"].mean(axis=1), lcohA_incr=Ai["lcoh"],
        H2A_div=Av["H2_y"].mean(axis=1), H2A_lic_div=Alv["H2_y"].mean(axis=1), lcohA_div=Av["lcoh"],
        gen_div=Gv[:, IH:].mean(axis=1),
        H2B_comp=Bc["H2_y"].mean(axis=1), lcohB_comp=Bc["lcoh"],
    )


# ---------------------------------------------------------------- run
def main():
    SITE_KEYS = ["H2A", "H2A_cc", "H2A_singh", "H2A_lic", "H2A_pot", "H2B", "lcohA", "lcohA_cc",
                 "lcohA_singh", "lcohB", "ciA", "ciA_cc", "ciA_cc_slip", "ciB", "oppB", "avoidA", "avoid_t",
                 "gen", "rec", "gen_bey", "eta", "eta0", "E", "Pel", "co2", "co2_cc", "S", "Qfeed",
                 "capA", "capB", "G_2024", "H2A_incr", "lcohA_incr", "H2A_div", "H2A_lic_div",
                 "lcohA_div", "gen_div", "H2B_comp", "lcohB_comp"]
    # the split variant writes only national totals, so it keeps no per-site draws
    acc = {} if SPLIT else {k: np.empty((NDRAW, NS), dtype=np.float32) for k in SITE_KEYS}
    techA = np.zeros((NDRAW, NS), dtype=np.int8)
    techB = np.zeros((NDRAW, NS), dtype=np.int8)
    PARAM_KEYS = ["u_eta", "F", "u_E", "eta_ref", "s_up", "s_eng", "s_mtur", "sec", "nR", "z_cREF",
                  "cEL", "nE", "fL", "i", "ccc", "tns", "q_regen", "e_cc", "pe", "e_aux", "n_up",
                  "s_burn", "u_psa", "co2_ts"]
    params, nat = [], []
    t0 = time.time()
    act_mask = active
    lic = cat["licence_status"].to_numpy()
    GROUPS = {"licensed": lic == "licensed", "expired": lic == "expired",
              "notreported": lic == "not reported", "closed": lic == "closed"}
    life = cat["remaining_life_bin"].astype(str).to_numpy()
    cy = cat["closure_year"].to_numpy(dtype=float)
    STATUS = {"gt15": active & (life == ">15 yr"), "l10_15": active & (life == "10-15 yr"),
              "l5_10": active & (life == "5-10 yr"), "l2_5": active & (life == "2-5 yr"),
              "l0_2": active & (life == "0-2 yr"), "expired": active & (lic == "expired"),
              "notrep": active & (lic == "not reported"),
              "c2020": ~active & (cy >= 2020) & (cy <= 2024), "c2015": ~active & (cy >= 2015) & (cy <= 2019),
              "c2011": ~active & (cy >= 2011) & (cy <= 2014)}


    def wmedian(x, w):
        ok = np.isfinite(x) & (w > 0)
        if not ok.any():
            return np.nan
        o = np.argsort(x[ok])
        cw = np.cumsum(w[ok][o])
        return float(x[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])


    for d in range(NDRAW):
        p = draw_params()
        r_all = one_draw(p)
        for k in acc:
            acc[k][d] = r_all[k]
        techA[d], techB[d] = r_all["techA"], r_all["techB"]
        row = {k: p[k] for k in PARAM_KEYS}
        row.update({f"u_k{c + 1}": p["u_k"][c] for c in range(4)})
        row.update({f"l0m{c + 1}": p["l0m"][c] for c in range(4)})
        params.append(row)
        # the national aggregates cover the sites active in 2024, which are the scope of the article; every site keeps its own results, and the closed sites are summed apart so that their share of the potential can be reported
        r = {k: (v[act_mask] if isinstance(v, np.ndarray) and v.shape == (NS,) else v)
             for k, v in r_all.items()}
        H2A, H2B, lA, lB = r["H2A"], r["H2B"], r["lcohA"], r["lcohB"]
        n = {"H2A_kt": H2A.sum() / 1e6, "H2A_cc_kt": r["H2A_cc"].sum() / 1e6,
             "H2A_singh_kt": r["H2A_singh"].sum() / 1e6, "H2A_lic_kt": r["H2A_lic"].sum() / 1e6,
             "H2A_pot_kt": r["H2A_pot"].sum() / 1e6, "H2B_kt": H2B.sum() / 1e6,
             "H2A_all_kt": r_all["H2A"].sum() / 1e6, "H2A_closed_kt": r_all["H2A"][~act_mask].sum() / 1e6,
             "H2B_all_kt": r_all["H2B"].sum() / 1e6, "H2B_closed_kt": r_all["H2B"][~act_mask].sum() / 1e6,
             "H2A_lic_all_kt": r_all["H2A_lic"].sum() / 1e6,
             "gen_all_Mm3": r_all["gen"].sum() / 1e6, "rec_all_Mm3": r_all["rec"].sum() / 1e6,
             "gen2024_all_kt": r_all["G_2024"].sum() * RHO_CH4 / 1e6,
             "gen_Mm3": r["gen"].sum() / 1e6, "rec_Mm3": r["rec"].sum() / 1e6,
             "gen_kt": r["gen"].sum() * RHO_CH4 / 1e6, "rec_kt": r["rec"].sum() * RHO_CH4 / 1e6,
             "rec_beyond_share": r["gen_bey"].sum() / max(r["rec"].sum(), 1e-9),
             "gen2024_kt": r["G_2024"].sum() * RHO_CH4 / 1e6,
             "G_dj_kt": r["G_dj"] * RHO_CH4 / 1e6, "avoid_Mt": np.nansum(r["avoid_t"]) / 1e6,
             "co2_stored_Mt": float(np.nansum(r["co2_cc"])) / 1e9,
             "Pel_MW": float(np.nansum(r["Pel"])) / 1e3,
             "B_feasible_sites": int(np.isfinite(lB).sum())}
        n["H2A_incr_kt"] = r["H2A_incr"].sum() / 1e6
        n["H2A_div_kt"] = r["H2A_div"].sum() / 1e6
        n["H2A_lic_div_kt"] = r["H2A_lic_div"].sum() / 1e6
        n["gen_div_Mm3"] = r["gen_div"].sum() / 1e6
        n["H2B_comp_kt"] = r["H2B_comp"].sum() / 1e6
        n["A_below_3_incr"] = float(r["H2A_incr"][np.nan_to_num(r["lcohA_incr"], nan=np.inf) < 3.0].sum()
                                    / max(r["H2A_incr"].sum(), 1e-9))
        n["A_below_3_div"] = float(r["H2A_div"][np.nan_to_num(r["lcohA_div"], nan=np.inf) < 3.0].sum()
                                   / max(r["H2A_div"].sum(), 1e-9))
        for thr in (2.0, 3.0, 4.0, 5.0):
            n[f"A_below_{thr:g}"] = float(H2A[np.nan_to_num(lA, nan=np.inf) < thr].sum() / max(H2A.sum(), 1e-9))
            n[f"B_below_{thr:g}"] = float(H2B[np.nan_to_num(lB, nan=np.inf) < thr].sum() / max(H2B.sum(), 1e-9))
        w = H2A > 0
        n["ciA_w"] = float(np.nansum(r["ciA"][w] * H2A[w]) / max(H2A[w].sum(), 1e-9))
        n["ciA_cc_w"] = float(np.nansum(r["ciA_cc"][w] * r["H2A_cc"][w]) / max(r["H2A_cc"][w].sum(), 1e-9))
        n["ciA_cc_slip_w"] = float(np.nansum(r["ciA_cc_slip"][w] * r["H2A_cc"][w]) / max(r["H2A_cc"][w].sum(), 1e-9))
        n["ciA_max"] = float(np.nanmax(np.where(w, r["ciA"], np.nan)))
        wb = H2B > 0
        n["ciB_w"] = float(np.nansum(r["ciB"][wb] * H2B[wb]) / max(H2B[wb].sum(), 1e-9))
        n["avoidA_w"] = float(np.nansum(r["avoid_t"]) * 1000.0 / max(H2A.sum(), 1e-9))
        both = np.isfinite(lA) & np.isfinite(lB)
        n["A_cheaper_share"] = float(H2A[both & (lA < lB)].sum() / max(H2A[both].sum(), 1e-9))
        n["lcohA_wmed"] = wmedian(lA, H2A)
        n["lcohB_wmed"] = wmedian(lB, H2B)
        n["lcohA_cc_wmed"] = wmedian(r["lcohA_cc"], r["H2A_cc"])
        n["lcohA_incr_wmed"] = wmedian(r["lcohA_incr"], r["H2A_incr"])
        n["lcohA_div_wmed"] = wmedian(r["lcohA_div"], r["H2A_div"])
        n["lcohB_comp_wmed"] = wmedian(r["lcohB_comp"], r["H2B_comp"])
        for gname, gmask in GROUPS.items():
            n[f"H2A_{gname}_kt"] = r_all["H2A"][gmask].sum() / 1e6
            n[f"H2A_lic_{gname}_kt"] = r_all["H2A_lic"][gmask].sum() / 1e6
        for gname, gmask in STATUS.items():
            n[f"st_{gname}_kt"] = r_all["H2A"][gmask].sum() / 1e6
            n[f"st_lic_{gname}_kt"] = r_all["H2A_lic"][gmask].sum() / 1e6
        nat.append(n)
        if (d + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  draw {d + 1}/{NDRAW}  {el:.0f} s", flush=True)

    nat = pd.DataFrame(nat)
    params = pd.DataFrame(params)
    if SPLIT:
        nat.to_csv(RES / "mc_national_split.csv", index=False, float_format="%.6g")
        print(nat[["H2A_kt", "lcohA_wmed", "A_below_3"]].quantile([0.1, 0.5, 0.9]).round(3).to_string())
        print(f"split variant done in {time.time() - t0:.0f} s")
        return
    nat.to_csv(RES / "mc_national.csv", index=False, float_format="%.6g")
    params.to_csv(RES / "mc_params.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------------------- site summary
    out = cat.copy()
    out["zone"] = zone + 1
    out["share_measured"] = share_measured
    out["L0_mean"] = (Wc.sum(axis=2) @ L0C) / np.maximum(W.sum(axis=1), 1e-9)
    out["wip_2025_t"] = CUM[:, int(np.searchsorted(DEP, 2025))]
    out["eta_c_basis"] = np.where(has_t, "measured capture", np.where(has_f, "confirmed field", "none"))
    for k in SITE_KEYS:
        a = acc[k].astype(float)
        fin = np.isfinite(a)
        allnan = ~fin.any(axis=0)
        with np.errstate(invalid="ignore"):
            q = np.nanpercentile(np.where(fin, a, np.nan), [10, 50, 90], axis=0) if fin.any() else None
        for lab, j in (("p10", 0), ("p50", 1), ("p90", 2)):
            out[f"{k}_{lab}"] = np.where(allnan, np.nan, q[j])
    names = np.array(GEN_NAMES + ["none"])
    out["techA_mode"] = [names[np.bincount(techA[:, j].astype(int), minlength=4).argmax()] for j in range(NS)]
    tb = np.where(techB < 0, 4, techB).astype(int)
    out["techB_mode"] = [names[np.bincount(tb[:, j], minlength=5).argmax()] for j in range(NS)]
    out["B_feasible_share"] = np.isfinite(acc["lcohB"]).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lB = acc["lcohB"].astype(float)
        for key, col in (("lcohA", "routeA_share"), ("lcohA_singh", "routeA_singh_share")):
            lA = acc[key].astype(float)
            both = np.isfinite(lA) & np.isfinite(lB)
            nb = both.sum(axis=0)
            out[col] = np.where(nb > 0, (both & (lA < lB)).sum(axis=0) / np.maximum(nb, 1), np.nan)
    out.to_csv(RES / "site_results.csv", float_format="%.6g")
    np.savez_compressed(RES / "mc_site_draws.npz", site_id=np.array(SITES),
                        **{k: acc[k] for k in ("H2A", "lcohA", "H2B", "lcohB", "H2A_cc", "lcohA_cc")})
    meta = {"seed": SEED, "ndraw": NDRAW, "sites": NS, "years": [Y0, Y1], "deposits_from": DEP0,
            "gwp": GWP, "oxidation": OX, "availability": AV, "capture_fraction": CAPT,
            "E_small": E_SMALL, "E_large": E_LARGE, "E_singh": E_SINGH,
            "R_psa_small": R_PSA_SMALL, "R_psa_large": R_PSA_LARGE, "co2_ts": CO2_TS,
            "Q_small_m3h": Q_SMALL, "Q_large_m3h": Q_LARGE, "site_class": SITE_CLASS,
            "wip_year_for_wells": WIP_YEAR, "lag_yr": LAG, "sections": NSEC, "s_burn": S_BURN,
            "diversion": [DIV_LEVEL, DIV_Y0, DIV_FULL], "e_comp_kwh_kg": E_COMP,
            "runtime_s": round(time.time() - t0, 1)}
    (RES / "model_meta.json").write_text(json.dumps(meta, indent=1))
    q = nat.quantile([0.05, 0.5, 0.95]).T
    print(q.round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
