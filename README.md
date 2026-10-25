# Landfill-gas hydrogen atlas of Colombia

[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey.svg)](inputs/LICENSE)

This repository holds the code and the input data of the article *A Site-Resolved Atlas of Potential, Cost and Carbon Intensity for Green Hydrogen from Landfill Gas in Colombia*, submitted to *Hydrogen* (MDPI). The code estimates, for each of the 248 disposal sites active in Colombia in 2024, the methane that the deposited municipal solid waste (MSW) generates and that a collection field would recover between 2026 and 2040, the hydrogen that two routes would produce from it, their levelized cost of hydrogen (LCOH) and their cradle-to-gate carbon intensity (CI), and it propagates the uncertainty of every input through a Monte Carlo simulation of 10,000 realizations.

The two routes are

- **Route A**, catalytic steam reforming of the upgraded landfill gas (LFG), with pressure swing adsorption (PSA) and, as a variant, capture and storage of the process CO2;
- **Route B**, alkaline electrolysis powered by the least-cost LFG generator that each site can support.

From the inputs of `inputs/`, the scripts compute every result reported in the article and in its Supplementary Materials, and write them as CSV, JSON and NPZ files, together with the values quoted in the text (`results/numbers_*.csv`).

## Contents

| Path | What it holds |
|---|---|
| `run_all.py` | Runs the numbered scripts of `scripts/` in their order |
| `scripts/` | The scripts, numbered in the order in which they run, from the registry to the national results and the variants |
| `inputs/` | The public records and the tables derived from them that the code reads, with their sources in `inputs/PROVENANCE.md`, their license in `inputs/LICENSE` and their SHA-256 checksums in `inputs/checksums.sha256` |
| `inputs/records/` | Records of recovered methane, national disposal and site registers |
| `inputs/composition/` | Measured MSW compositions of 56 municipalities and the departmental profiles of the UPME |
| `docs/` | Notes on the modeling choices (`pipeline.md`), on the composition of the MSW (`composition.md`) and the columns of every result file (`data_dictionary.md`) |

## Requirements

- Python 3.11 or later (the results of the article were obtained with Python 3.13)
- pandas, NumPy, SciPy, statsmodels, GeoPandas (with Shapely, pyogrio and pyproj), openpyxl and Cantera; the exact versions are pinned in `requirements.txt` and `environment.yml`

With conda,

```bash
conda env create -f environment.yml
conda activate lfg-h2-atlas
```

or with pip, in a virtual environment,

```bash
python -m pip install -r requirements.txt
```

## Data

The inputs are public records or tables derived from them, namely the final-disposal registry of the Superintendency of Residential Public Services (SSPD), the population projections and the political-administrative division of the National Administrative Department of Statistics (DANE), the municipal boundaries of the Agustín Codazzi Geographic Institute (IGAC), the precipitation normals of the Institute of Hydrology, Meteorology and Environmental Studies (IDEAM), the MSW characterizations of the municipal waste management plans (PGIRS) and of the Mining and Energy Planning Unit (UPME), and the records of recovered methane of the operator of the Doña Juana landfill, of the Clean Development Mechanism (CDM) and of the SSPD. `inputs/PROVENANCE.md` lists the source, access date and changes of each file. The integrity of the inputs can be checked from the root of the repository with

```bash
sha256sum -c inputs/checksums.sha256
```

## Running the pipeline

From the root of the repository,

```bash
python run_all.py         # 10,000 realizations, as in the article
python run_all.py 500     # a quicker check with fewer realizations
```

The scripts create the folder `results/` and write their outputs there; each script can also be run on its own, from the root of the repository, once the scripts that precede it have run.

The seed is fixed (20260910), so two runs with the same number of realizations give identical outputs, and a run with 10,000 realizations reproduces the article. On a laptop, the chain takes about one hour, most of it in `scripts/20_variants.py`; with 500 realizations, it takes a few minutes. No step needs network access.

| Step | Script | Writes |
|---|---|---|
| 1 | `scripts/01_pop.py` | `results/population_municipal.csv` |
| 2 | `scripts/02_site_catalog.py` | `results/site_catalog.csv`, `site_crosswalk.csv`, `site_rows_assigned.csv`, `site_status_summary.csv` |
| 3 | `scripts/03_build_waste.py` | `results/waste_municipality_site_year.csv`, `waste_site_year.csv`, `waste_pools.csv`, `waste_checks.json` |
| 4 | `scripts/04_build_composition.py` | `results/composition_municipality.csv`, `composition_site.csv`, `composition_stats.json` |
| 5 | `scripts/05_precip_zones.py` | `results/site_precipitation.csv`, `precip_stats.json`, `numbers_precip.csv` |
| 6 | `scripts/06_geo.py` | `results/mun.gpkg`, `dept.gpkg`, `mun_points.csv` |
| 7 | `scripts/07_capex_data.py` | `results/capex_raw.csv`, the reformer cost record transcribed from the literature |
| 8 | `scripts/08_capex_regression.py` | `results/capex_normalised.csv`, `capex_final.json`, `numbers_capex.csv` |
| 9 | `scripts/09_capex_checks.py` | `results/capex_checks.json`, prediction intervals and alternative fits |
| 10 | `scripts/10_reformer_balance.py` | `results/reformer_balance.json`, `reformer_balance_cases.csv` |
| 11 | `scripts/11_lfg_h2_model.py NDRAW` | `results/site_results.csv`, `mc_national.csv`, `mc_params.csv`, `mc_site_draws.npz`, `model_meta.json` |
| 12 | `scripts/12_lfg_h2_model_split.py NDRAW` | `results/mc_national_split.csv` |
| 13 | `scripts/13_validation.py` | `results/validation.json` |
| 14 | `scripts/14_consistency.py NDRAW` | `results/consistency_sites.csv`, `consistency_dj_annual.csv`, `consistency_gen_draws.npz` |
| 15 | `scripts/15_defaults.py` | `results/numbers_defaults.csv`, quantities at the default parameter values |
| 16 | `scripts/16_sensitivity.py` | `results/sensitivity.csv`, partial rank correlation coefficients |
| 17 | `scripts/17_aggregate.py` | `results/national_results.json`, `numbers_results.csv` |
| 18 | `scripts/18_balances.py` | `results/balances.csv` |
| 19 | `scripts/19_profiles.py` | `results/profiles.csv` |
| 20 | `scripts/20_variants.py NDRAW` | `results/variants_national.csv`, `variants_years.npz` |

## Model in brief

- **Sites and deposits.** The registry rows of 2010 to 2024 are harmonized into canonical sites, and the deposits of each site are back-cast before the first observed year and projected to 2040 from the per-capita disposal of 2020 to 2024, with every projected deposit split at the licensed date into a licensed and a beyond-license layer.
- **Methane.** A first-order decay model with four decay categories and five precipitation zones of the Colombia Landfill Gas Model, a methane correction factor by site class, and a collection efficiency that is triangular (0.41, 0.60, 0.71) at sanitary landfills, with one quantile shared by all sites; a variant follows the cover stage of each annual deposit.
- **Route A.** The overall reformer efficiency is interpolated in the logarithm of the design flow between 0.60 at 1.5 t/d and 0.76 at 216 t/d, the PSA recovery between 0.78 and 0.89, and the fuel-to-feed ratio of the reformer follows from both and from the hydrogen formed per mole of feed; the auxiliary electricity comes from the least-cost LFG generator.
- **Route B.** The least-cost feasible generator of the LFGcost-Web correlations of the US EPA feeds an alkaline electrolyzer.
- **Costs.** Discounted cash flow over 2026 to 2040, in 2023 USD escalated with the chemical engineering plant cost index, with the reformer capital from a pooled, scope-harmonized fit of 21 published estimates (exponent 0.806).
- **Carbon.** Biogenic CO2 is neutral, so that the CI comes from the methane lost in upgrading and in the generators and burners; capture credits only the concentrated process CO2, net of the emissions of its transport and storage.
- **Uncertainty.** Every input of Table 18 of the article is drawn per realization, and the national figures are the medians and the 10th and 90th percentiles of the totals over the active sites.

## Citation

If you use this code or its input data, please cite the article and this record, and the original sources listed in `inputs/PROVENANCE.md`; `CITATION.cff` holds the references in machine-readable form.

> Otero Meza, D.D.; Vidal Barrero, F.; Cabello Eras, J.J.; Solarte Toro, J.C.; Cardona Alzate, C.A. A Site-Resolved Atlas of Potential, Cost and Carbon Intensity for Green Hydrogen from Landfill Gas in Colombia. *Hydrogen* (under review).

> Otero Meza, D.D. *Landfill-gas hydrogen atlas of Colombia*. Zenodo, 2026.

## License

The code is released under the MIT License (see `LICENSE`). The tables prepared by the authors in `inputs/` are released under the Creative Commons Attribution 4.0 International license (see `inputs/LICENSE`), and the public records redistributed there remain subject to the terms of their publishers, listed in `inputs/PROVENANCE.md`.

## Contact

Daniel D. Otero Meza, INGEBIR Research Group, Department of Agricultural Engineering, University of Sucre, Sincelejo, Colombia (daniel.otero@unisucre.edu.co).
