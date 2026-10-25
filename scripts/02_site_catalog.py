"""Canonical catalogue of the disposal sites in the Colombian registry, 2010-2024.

Each row of the registry is one served municipality delivering to one site in one year. The same site appears under several spellings, and some rows carry the code or the department of the municipality served instead of those of the site. This script groups the rows into canonical sites and gives each one its status in the base year, its licensed end of life, its remaining life at the start of the assessment horizon and, where documented, its opening year.

Grouping rules, in the order they are applied
 1. Names are normalised (accents, punctuation, case) and stripped of generic words (relleno sanitario, parque ambiental, regional, ...) to a core name. DANE code 05540 is recoded to 05400, as in the regulator's corrected series.
 2. Curated overrides (inputs/site_overrides.csv) merge the variants of the large regional sites, including rows filed under a served department.
 3. Row-level fixes (inputs/site_row_fixes.csv) reassign documented site-code errors of single municipality-years.
 4. Within a department, a non-generic core name recorded under several site codes takes its dominant code when that code carries at least 70 % of the name's tonnage. Generic names (botadero, municipal, sin nombre, ...) keep the code they were filed under.
 5. The few rows without a site name or code take the site of the same served municipality in the nearest reported year.

Status in the base year (2024)
    active   the site received waste in 2024
    closed   last deposit before 2024; closure year = last reported year + 1

Licensed end of life
    The date the operator reported to the regulator in the site's last reported year (field lifespan_end_date of the registry, tonnage-weighted mode). Remaining life is counted from 1 January 2026, the first year of the assessment horizon. For an active site the licence is
        licensed        date after 1 January 2026
        expired         date on or before 1 January 2026, site still receiving
        not reported    no date
    The dates are cross-checked against the SSPD site inventory (inputs/records/sspd_site_inventory.csv), built from the same regulator reports.

Opening year
    Curated for the large sites from the CDM project documents and the SUI register (inputs/site_facts.csv); for the rest, the SUI opening date when it is not later than the first year the site appears in the registry.

Outputs
    results/site_crosswalk.csv    raw (department, site code, site name) -> site_id
    results/site_catalog.csv      one row per canonical site
    results/site_status_summary.csv  counts and tonnage by status and licence

Run from the project root:
    python scripts/02_site_catalog.py
"""

import pathlib
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
MA_DATA = INPUTS / "records"

BASE_YEAR = 2024
HORIZON_START = pd.Timestamp("2026-01-01")
DOMINANT_SHARE = 0.70
RECODE_DANE = {"05540": "05400"}

STRIP = ("RELLENO SANITARIO", "RELLENO SANITARO", "RELLENO SANTIARIO",
         "PARQUE AMBIENTAL", "PARQUE ECOLOGICO", "PARQUE TECNOLOGICO AMBIENTAL",
         "PARQUE INDUSTRIAL DE RESIDUOS SOLIDOS", "REGIONAL",
         "SITIO DE DISPOSICION FINAL", "RELLENO")
GENERIC = re.compile(
    r"^(|BOTADERO|BOTADERO A CIELO ABIERTO|BOTADERO MUNICIPAL|BASURERO|"
    r"BASURERO MUNICIPAL|MUNICIPAL|NO POSEE|SIN NOMBRE|CELDA|CELDA TRANSITORIA|"
    r"CELDA TRANSITORIA DE DISPOSICION DE RESIDUOS SOLIDOS|CELDA DE CONTINGENCIA|"
    r"ENTERRAMIENTO|PLANTA DE MANEJO INTEGRAL|PLANTA DE RESIDUOS SOLIDOS|"
    r"CUERPO DE AGUA.*|QUEMA.*|NO REPORTA.*)$")


def norm(s):
    """Upper case, no accents, letters and digits only, single spaces."""
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s)).strip()


def core(s):
    """Normalised name without the generic words that precede most site names."""
    s = " " + norm(s) + " "
    for w in STRIP:
        s = s.replace(" " + w + " ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    t = re.sub(r"^(DEL|DE|Y)\s+", "", s)
    return t if t else s


def parse_date(s):
    """Registry dates are mostly month/day/year, a few are day/month/year."""
    if pd.isna(s):
        return pd.NaT
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def load_registry():
    lf = pd.read_csv(INPUTS / "landfilling_site_code_imputed.csv", dtype=str)
    lf["tons"] = pd.to_numeric(lf["avg_tons_per_year"], errors="coerce")
    lf["year"] = pd.to_numeric(lf["year"], errors="coerce").astype(int)
    lf = lf[lf["tons"] > 0].copy().reset_index(drop=True)
    lf["served_dane_code"] = lf["served_dane_code"].str.zfill(5)
    code = lf["site_dane_code"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    lf["site_dane_code"] = code.replace(RECODE_DANE)
    lf["dept_n"] = lf["site_department"].map(norm)
    lf["core"] = lf["site_name"].map(core)
    # a site code without a department label takes the department of the code
    lab = lf[(lf["dept_n"] != "") & lf["site_dane_code"].notna()]
    dept_of = lab.groupby(lab["site_dane_code"].str[:2])["dept_n"].agg(
        lambda x: x.mode().iloc[0])
    miss = (lf["dept_n"] == "") & lf["site_dane_code"].notna()
    lf.loc[miss, "dept_n"] = lf.loc[miss, "site_dane_code"].str[:2].map(dept_of)
    # a named site without a code is taken to lie in the municipality it serves
    named = lf["site_dane_code"].isna() & (lf["core"] != "")
    lf.loc[named, "site_dane_code"] = lf.loc[named, "served_dane_code"]
    miss = (lf["dept_n"].fillna("") == "") & lf["site_dane_code"].notna()
    lf.loc[miss, "dept_n"] = lf.loc[miss, "site_dane_code"].str[:2].map(dept_of)
    lf["eol"] = lf["lifespan_end_date"].map(parse_date)
    lf.loc[lf["eol"] > pd.Timestamp("2075-12-31"), "eol"] = pd.NaT
    return lf


def assign_sites(lf):
    ov = pd.read_csv(INPUTS / "site_overrides.csv", dtype=str).fillna("")
    fx = pd.read_csv(INPUTS / "site_row_fixes.csv", dtype=str).fillna("")
    for c in ("canon_dept", "canon_dane", "canon_core", "display_name", "rule"):
        lf[c] = pd.Series([None] * len(lf), dtype=object)

    for o in ov.itertuples(index=False):
        m = (lf["dept_n"].fillna("") == o.dept_n) & lf["core"].str.match(o.core_regex) \
            & lf["canon_dane"].isna()
        lf.loc[m, ["canon_dept", "canon_dane", "canon_core", "display_name", "rule"]] = \
            [o.canon_dept, o.canon_dane, o.canon_core, o.display_name, "override"]

    for f in fx.itertuples(index=False):
        m = (lf["served_dane_code"] == f.served_dane_code) & (lf["year"] == int(f.year)) \
            & lf["core"].str.match(f.from_core_regex)
        disp = ov.loc[(ov.canon_dane == f.to_dane) & (ov.canon_core == f.to_core),
                      "display_name"]
        lf.loc[m, ["canon_dept", "canon_dane", "canon_core", "display_name", "rule"]] = \
            [f.to_dept, f.to_dane, f.to_core,
             disp.iloc[0] if len(disp) else None, "row fix"]

    rest = lf["canon_dane"].isna() & lf["site_dane_code"].notna()
    for (dept, cname), g in lf[rest].groupby(["dept_n", "core"]):
        idx = g.index
        if GENERIC.match(cname):
            lf.loc[idx, "canon_dane"] = g["site_dane_code"]
            lf.loc[idx, "rule"] = "own code (generic name)"
        else:
            share = g.groupby("site_dane_code")["tons"].sum()
            share = share / share.sum()
            if share.max() >= DOMINANT_SHARE:
                lf.loc[idx, "canon_dane"] = share.idxmax()
                lf.loc[idx, "rule"] = "dominant code" if len(share) > 1 else "single code"
            else:
                lf.loc[idx, "canon_dane"] = g["site_dane_code"]
                lf.loc[idx, "rule"] = "own code (no dominant code)"
        lf.loc[idx, "canon_dept"] = dept
        lf.loc[idx, "canon_core"] = cname if cname else "SIN NOMBRE"

    # rows with neither a name nor a code: nearest reported year of the same served municipality
    orphan = lf.index[lf["canon_dane"].isna()]
    for i in orphan:
        r = lf.loc[i]
        cand = lf[(lf["served_dane_code"] == r["served_dane_code"]) & lf["canon_dane"].notna()]
        if cand.empty:
            lf.loc[i, ["canon_dept", "canon_dane", "canon_core", "rule"]] = \
                [r["dept_n"], r["served_dane_code"], "NO IDENTIFICADO", "unidentified"]
            continue
        dy = (cand["year"] - r["year"]).abs() * 2 - (cand["year"] > r["year"])
        best = cand.loc[dy.idxmin()]
        lf.loc[i, ["canon_dept", "canon_dane", "canon_core", "display_name", "rule"]] = \
            [best["canon_dept"], best["canon_dane"], best["canon_core"],
             best["display_name"], "nearest-year site of the served municipality"]

    lf["site_id"] = lf["canon_dane"] + "|" + lf["canon_core"]
    return lf


def wmode(values, weights):
    """Tonnage-weighted mode of a series, ignoring missing values."""
    d = pd.DataFrame({"v": values, "w": weights}).dropna()
    if d.empty:
        return np.nan
    return d.groupby("v")["w"].sum().idxmax()


def display_from(name):
    s = re.sub(r"^\s*(RELLENO SANITARIO|RELLENO)\s+", "", str(name).strip(), flags=re.I)
    s = re.sub(r"^(Y|REGIONAL)\s+", "", s, flags=re.I)
    return s.title().replace(" De ", " de ").replace(" Del ", " del ").replace(
        " La ", " la ").replace(" Los ", " los ").replace(" Las ", " las ").replace(
        " El ", " el ").replace(" Y ", " y ")


def build_catalog(lf):
    g = lf.groupby("site_id")
    cat = pd.DataFrame({
        "dept": g["canon_dept"].first(),
        "dane": g["canon_dane"].first(),
        "core": g["canon_core"].first(),
        "first_year": g["year"].min(),
        "last_year": g["year"].max(),
        "n_years": g["year"].nunique(),
        "tons_2010_2024": g["tons"].sum(),
        "n_name_variants": g["site_name"].nunique(),
        "n_codes_merged": g["site_dane_code"].nunique(),
    })
    cat.index.name = "site_id"
    cat["rule"] = g["rule"].agg(lambda x: ", ".join(sorted(set(x.dropna()))))
    top_name = (lf.groupby(["site_id", "site_name"])["tons"].sum().reset_index()
                  .sort_values("tons").groupby("site_id")["site_name"].last())
    cat["site_name"] = top_name
    disp = g["display_name"].agg(lambda x: x.dropna().iloc[0] if x.notna().any() else None)
    cat["display_name"] = disp.where(disp.notna(), cat["site_name"].map(display_from))
    cat["site_type"] = g.apply(lambda d: wmode(d["site_type"], d["tons"]), include_groups=False)

    # municipality of the site code
    mun = pd.concat([
        lf[["site_dane_code", "site_municipality"]].rename(
            columns={"site_dane_code": "code", "site_municipality": "mun"}),
        lf[["served_dane_code", "served_municipality"]].rename(
            columns={"served_dane_code": "code", "served_municipality": "mun"}),
    ]).dropna()
    mun_of = mun.groupby("code")["mun"].agg(lambda x: x.mode().iloc[0])
    cat["municipality"] = cat["dane"].map(mun_of)
    # generic names are identified by their municipality
    gen = cat["core"].map(lambda c: bool(GENERIC.match(c)) or c in ("SIN NOMBRE", "NO IDENTIFICADO"))
    lab = cat.loc[gen, "core"].map(lambda c: {"SIN NOMBRE": "Unnamed site", "": "Unnamed site",
                                              "NO IDENTIFICADO": "Unidentified site",
                                              "MUNICIPAL": "Municipal site"}.get(c, c.capitalize()))
    cat.loc[gen, "display_name"] = lab + ", " + cat.loc[gen, "municipality"].fillna("").str.title()

    # base-year and last-year figures
    last = lf[lf["year"] == lf["site_id"].map(cat["last_year"])]
    gl = last.groupby("site_id")
    cat["tons_last_year"] = gl["tons"].sum()
    cat["municipalities_served_last_year"] = gl["served_dane_code"].nunique()
    cat["site_type_last_year"] = gl.apply(lambda d: wmode(d["site_type"], d["tons"]),
                                          include_groups=False)
    cat["tons_2024"] = lf[lf["year"] == BASE_YEAR].groupby("site_id")["tons"].sum()
    cat["tons_2024"] = cat["tons_2024"].fillna(0.0)
    cat["tpd_2024"] = cat["tons_2024"] / 365.0
    cat["regional"] = cat["municipalities_served_last_year"] > 1

    cat["status_2024"] = np.where(cat["last_year"] == BASE_YEAR, "active", "closed")
    cat["closure_year"] = np.where(cat["status_2024"] == "closed", cat["last_year"] + 1, np.nan)

    # licensed end of life: date reported in the last reported year
    eol = gl.apply(lambda d: wmode(d["eol"], d["tons"]), include_groups=False)
    cat["licence_end"] = pd.to_datetime(eol)
    rem = (cat["licence_end"] - HORIZON_START).dt.days / 365.25
    active = cat["status_2024"] == "active"
    cat["remaining_life_2026"] = np.where(active, rem, np.nan)
    cat["licence_status"] = np.select(
        [~active, cat["licence_end"].isna(), rem <= 0.0],
        ["closed", "not reported", "expired"], "licensed")
    bins = [-np.inf, 0.0, 2.0, 5.0, 10.0, 15.0, np.inf]
    lab = ["expired", "0-2 yr", "2-5 yr", "5-10 yr", "10-15 yr", ">15 yr"]
    cat["remaining_life_bin"] = pd.cut(cat["remaining_life_2026"], bins, labels=lab).astype(object)
    cat.loc[active & cat["licence_end"].isna(), "remaining_life_bin"] = "not reported"
    cat.loc[~active, "remaining_life_bin"] = "closed"
    return cat


class Resolver:
    """Map an outside record (department, code, name) to a canonical site."""

    def __init__(self, lf, cat):
        self.ov = pd.read_csv(INPUTS / "site_overrides.csv", dtype=str).fillna("")
        raw = lf.groupby(["site_dane_code", "core"])["site_id"].agg(
            lambda x: x.mode().iloc[0])
        self.raw = raw.to_dict()
        self.ids = set(cat.index)
        by = cat.reset_index().groupby(["dept", "core"])["site_id"].agg(list)
        self.by_dept = by.to_dict()

    def __call__(self, dept_n, dane, cname):
        for o in self.ov.itertuples(index=False):
            if dept_n == o.dept_n and re.match(o.core_regex, cname):
                return f"{o.canon_dane}|{o.canon_core}"
        if f"{dane}|{cname}" in self.ids:
            return f"{dane}|{cname}"
        if (dane, cname) in self.raw:
            return self.raw[(dane, cname)]
        cand = self.by_dept.get((dept_n, cname), [])
        return cand[0] if len(cand) == 1 else None


def crosscheck_inventory(cat, resolve):
    """Licence dates of the SSPD site inventory for the sites active in the base year."""
    inv = pd.read_csv(MA_DATA / "sspd_site_inventory.csv", dtype={"dane_site": str})
    inv = inv[inv["last_year"] == BASE_YEAR].copy()
    inv["dane_site"] = inv["dane_site"].str.zfill(5).replace(RECODE_DANE)
    inv["cat_id"] = [resolve(norm(d), c, core(n)) for d, c, n in
                     zip(inv["department"], inv["dane_site"], inv["site_name"])]
    inv["eol_ma"] = pd.to_datetime(inv["end_of_life"], errors="coerce")
    m = inv.dropna(subset=["cat_id"]).sort_values("tonnes_last_year").groupby("cat_id").last()
    return m["eol_ma"]


def opening_years(cat, resolve):
    """Curated opening years first, then the earliest plausible SUI date."""
    cat["opened_year"] = np.nan
    cat["opened_source"] = ""
    facts = INPUTS / "site_facts.csv"
    if facts.exists():
        f = pd.read_csv(facts, dtype={"site_id": str}).set_index("site_id")
        f = f[f["opened_year"].notna()]
        missing = sorted(set(f.index) - set(cat.index))
        if missing:
            raise SystemExit(f"site_facts.csv names unknown sites: {missing}")
        cat.loc[f.index, "opened_year"] = f["opened_year"].astype(float)
        cat.loc[f.index, "opened_source"] = f["opened_source"]
    sui = pd.read_csv(MA_DATA / "sui_landfill_sites.csv", dtype=str)
    sui = sui[sui["site_id"].notna() & sui["opened_on"].notna()].copy()
    parts = sui["site_id"].str.split("|", n=1)
    sui["dane"] = parts.str[0].str.zfill(5).replace(RECODE_DANE)
    sui["cat_id"] = [resolve(norm(d), c, core(n)) for d, c, n in
                     zip(sui["department"], sui["dane"], parts.str[1])]
    sui["yr"] = pd.to_datetime(sui["opened_on"], errors="coerce").dt.year
    sui = sui.dropna(subset=["cat_id", "yr"])
    sui["first_year"] = sui["cat_id"].map(cat["first_year"])
    sui = sui[(sui["yr"] >= 1950) & (sui["yr"] <= sui["first_year"])]
    m = sui.groupby("cat_id")["yr"].min()
    todo = cat["opened_year"].isna() & cat.index.isin(m.index)
    cat.loc[todo, "opened_year"] = m.reindex(cat.index[todo]).values
    cat.loc[todo, "opened_source"] = "SUI site register, earliest opened_on"
    # a site that first appears after the opening year of the registry is taken to have opened in that year; otherwise the opening is not constrained
    late = cat["opened_year"].isna() & (cat["first_year"] > 2010)
    cat.loc[late, "opened_year"] = cat.loc[late, "first_year"]
    cat.loc[late, "opened_source"] = "first year in the registry"
    cat["opened_source"] = cat["opened_source"].replace("", "not documented")
    return cat


def summarise(cat):
    rows = []
    for (st, lic), d in cat.groupby(["status_2024", "licence_status"]):
        rows.append({"status_2024": st, "licence_status": lic, "sites": len(d),
                     "tons_2010_2024_Mt": d["tons_2010_2024"].sum() / 1e6,
                     "tons_2024_Mt": d["tons_2024"].sum() / 1e6})
    s1 = pd.DataFrame(rows)
    rows = []
    for b, d in cat[cat["status_2024"] == "active"].groupby("remaining_life_bin"):
        rows.append({"status_2024": "active", "licence_status": "bin " + b, "sites": len(d),
                     "tons_2010_2024_Mt": d["tons_2010_2024"].sum() / 1e6,
                     "tons_2024_Mt": d["tons_2024"].sum() / 1e6})
    rows2 = []
    for y0, y1 in [(2011, 2014), (2015, 2019), (2020, 2024)]:
        d = cat[(cat["closure_year"] >= y0) & (cat["closure_year"] <= y1)]
        rows2.append({"status_2024": "closed", "licence_status": f"closure {y0}-{y1}",
                      "sites": len(d), "tons_2010_2024_Mt": d["tons_2010_2024"].sum() / 1e6,
                      "tons_2024_Mt": 0.0})
    return pd.concat([s1, pd.DataFrame(rows), pd.DataFrame(rows2)], ignore_index=True)


def main():
    lf = assign_sites(load_registry())
    cat = build_catalog(lf)
    resolve = Resolver(lf, cat)
    cat["licence_end_inventory"] = crosscheck_inventory(cat, resolve)
    cat = opening_years(cat, resolve)
    cat = cat.sort_values("tons_2010_2024", ascending=False)
    cat.index.name = "site_id"

    cw = (lf.groupby(["site_department", "site_dane_code", "site_name", "site_id", "rule"],
                     dropna=False)
            .agg(tons=("tons", "sum"), first_year=("year", "min"), last_year=("year", "max"),
                 rows=("tons", "size"))
            .reset_index())
    RESULTS.mkdir(exist_ok=True)
    cw.to_csv(RESULTS / "site_crosswalk.csv", index=False)
    lf[["served_dane_code", "year", "site_id", "tons"]].to_csv(
        RESULTS / "site_rows_assigned.csv", index=False)
    cols = ["display_name", "site_name", "dept", "dane", "municipality", "site_type",
            "site_type_last_year", "first_year", "last_year", "n_years", "status_2024",
            "closure_year", "tons_2010_2024", "tons_2024", "tpd_2024", "tons_last_year",
            "municipalities_served_last_year", "regional", "licence_end",
            "licence_end_inventory", "remaining_life_2026", "licence_status",
            "remaining_life_bin", "opened_year", "opened_source", "n_name_variants",
            "n_codes_merged", "rule"]
    cat[cols].to_csv(RESULTS / "site_catalog.csv", float_format="%.6g")
    summ = summarise(cat)
    summ.to_csv(RESULTS / "site_status_summary.csv", index=False, float_format="%.4f")

    tot = lf.groupby("year")["tons"].sum() / 1e6
    print(f"registry rows {len(lf):,}, raw site identities "
          f"{lf.groupby(['site_dane_code', 'site_name'], dropna=False).ngroups:,}, "
          f"canonical sites {len(cat):,}")
    print("rows by rule:", lf["rule"].value_counts().to_dict())
    print(f"active in {BASE_YEAR}: {(cat.status_2024 == 'active').sum()}, "
          f"closed: {(cat.status_2024 == 'closed').sum()}")
    print(summ.to_string(index=False))
    print("registry total, Mt/yr:", tot.round(2).to_dict())
    agree = cat[cat["licence_end"].notna() & cat["licence_end_inventory"].notna()]
    same = (agree["licence_end"] - agree["licence_end_inventory"]).abs().dt.days <= 31
    print(f"licence dates matched with the SSPD site inventory: {len(agree)}, "
          f"equal within a month: {int(same.sum())}")


if __name__ == "__main__":
    main()
