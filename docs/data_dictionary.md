# Data dictionary

The files described here are written into `results/` by the numbered scripts of `scripts/`, which `run_all.py` runs in order. The inputs are described in `inputs/PROVENANCE.md`.

## Conventions

- Hydrogen in kg/yr, except in columns ending in `_kt` (kt/yr); methane volumes in m3 at 15 °C and 101.325 kPa (`_Mm3`, million m3/yr); methane masses in t/yr or kt/yr as stated.
- LCOH in 2023 USD per kg of H2; CI and avoided methane in kg CO2e per kg of H2 (GWP of the Sixth Assessment Report of the IPCC); CAPEX in 2023 USD.
- Annual quantities are means of the 15 annual values between 2026 and 2040.
- Site results are percentiles over the 10,000 realizations, with the suffixes `_p10`, `_p50` and `_p90`.
- National results are totals or statistics over the 248 sites active in 2024, one value per realization, unless the column name contains `_all` (every site of the registry) or `_closed` (sites closed before 2024).
- Route A is catalytic steam reforming; Route B is electrolysis powered by landfill-gas electricity. `cc` marks the variant with capture of the process CO2, `lic` the licensed-only variant (no deposits after the licensed end of life), `singh` the conservative bound of the reformer efficiency (E = 0.35), `incr` the incremental variant (the gas already destroyed at four sites stays with the existing system), `div` the diversion of half of the food and garden waste by 2035 and `comp` the compression of the electrolytic H2 to 3 MPa.

## results/site_results.csv

One row per site of the national registry (1,868 rows).

| Column | Meaning |
|---|---|
| `site_id` | Canonical identifier, `DANE|NAME`, with the code of the host municipality and the normalized name |
| `display_name`, `site_name` | Name used in the article and name in the registry |
| `dept`, `dane`, `municipality` | Department, code and name of the host municipality |
| `site_type`, `site_type_last_year` | Site class in the registry (sanitary landfill, contingency cell, open dump, transitional cell, integral plant and others) over the whole record and in its last year |
| `first_year`, `last_year`, `n_years` | First and last year with reported disposal, 2010 to 2024, and number of years |
| `status_2024` | `active` or `closed` in 2024; the article covers the active sites |
| `closure_year` | Last year with disposal for the closed sites |
| `tons_2010_2024`, `tons_2024`, `tpd_2024`, `tons_last_year` | MSW received, t and t/d |
| `municipalities_served_last_year`, `regional` | Number of municipalities served and whether the site is regional |
| `licence_end`, `licence_end_inventory` | Licensed end of life in the 2024 registry and in the SSPD site inventory (`inputs/records/sspd_site_inventory.csv`), as a cross-check |
| `remaining_life_2026`, `licence_status`, `remaining_life_bin` | Licensed life remaining on 1 January 2026 (yr, negative if expired), license status and life group of Table 20 of the article |
| `opened_year`, `opened_source` | Opening year and its source |
| `n_name_variants`, `n_codes_merged`, `rule` | Number of registry names and municipal codes merged into the site, and the harmonization rule |
| `zone` | Precipitation zone of the Colombia Landfill Gas Model, from 1 (very wet, more than 2,000 mm/yr) to 5 (dry, less than 500 mm/yr) |
| `share_measured` | Share of the 2010–2024 intake with a measured characterization |
| `L0_mean` | Mean methane generation capacity of the deposited MSW, m3 CH4 per t |
| `wip_2025_t` | Waste in place at the end of 2025, t |
| `eta_c_basis` | Basis of the collection efficiency of the site class |
| `H2A`, `H2A_cc`, `H2A_singh`, `H2A_lic`, `H2B` | H2 of Route A in the base case, with capture, at the conservative efficiency bound and in the licensed-only variant, and H2 of Route B, kg/yr |
| `H2A_pot` | H2 of Route A if the plant used all the recovered methane, without the limit of its design flow, kg/yr |
| `lcohA`, `lcohA_cc`, `lcohA_singh`, `lcohB` | LCOH of the same configurations, USD/kg; empty where the route cannot be built |
| `ciA`, `ciA_cc`, `ciB` | CI of Route A without and with capture and of Route B, kg CO2e/kg |
| `ciA_cc_slip` | Part of `ciA_cc` due to methane slip, kg CO2e/kg |
| `oppB` | Value of the electricity that Route B consumes per kg of H2 if sold at the contract price, USD/kg |
| `avoidA`, `avoid_t` | Methane emission avoided by collection, kg CO2e per kg of Route A H2 and t CO2e/yr |
| `gen`, `rec`, `gen_bey` | Methane generated, recovered and recovered from deposits beyond the licensed date, m3/yr |
| `eta`, `eta0` | Collection efficiency and existing destruction as a fraction of the generation |
| `E` | Overall efficiency of the reformer island |
| `Pel` | Rating of the electrolyzer of Route B, kW |
| `co2`, `co2_cc` | Process CO2 of Route A and CO2 stored in the capture variant, kg/yr |
| `S`, `Qfeed` | Design output of the reformer, kg H2/d, and design flow of the cleanup and upgrading train, m3 LFG/h |
| `capA`, `capB` | CAPEX of Route A (collection, upgrading, reformer island and auxiliary generator) and of Route B, USD |
| `G_2024` | Methane generated in 2024, m3 |
| `H2A_incr`, `lcohA_incr`, `H2A_div`, `H2A_lic_div`, `lcohA_div`, `gen_div`, `H2B_comp`, `lcohB_comp` | Results of the incremental, diversion and compression variants |
| `techA_mode`, `techB_mode` | Modal auxiliary generator of Route A and modal generator of Route B (ENG, SENG, TUR, MTUR) |
| `B_feasible_share` | Share of the realizations in which Route B can be built |
| `routeA_share`, `routeA_singh_share` | Share of the realizations, among those in which both routes can be built, in which Route A has the lower LCOH, with the default and the bounding efficiency |

## results/mc_national.csv and results/mc_national_split.csv

One row per realization (10,000 rows). The split file repeats the ensemble with the scatter of the reformer CAPEX split into a common level and independent site deviations.

| Column | Meaning |
|---|---|
| `H2A_kt`, `H2A_cc_kt`, `H2A_singh_kt`, `H2A_lic_kt`, `H2A_pot_kt`, `H2B_kt` | National H2 of the configurations of `site_results.csv`, kt/yr |
| `H2A_all_kt`, `H2A_closed_kt`, `H2B_all_kt`, `H2B_closed_kt`, `H2A_lic_all_kt` | The same over every site of the registry and over the closed sites |
| `gen_Mm3`, `rec_Mm3`, `gen_kt`, `rec_kt`, `gen_all_Mm3`, `rec_all_Mm3` | Methane generated and recovered, Mm3/yr and kt/yr |
| `gen2024_kt`, `gen2024_all_kt` | Methane generated in 2024, kt |
| `rec_beyond_share` | Share of the recovered methane that comes from deposits beyond the licensed dates |
| `G_dj_kt` | Methane generated at Doña Juana, mean of 2021 to 2025, kt/yr |
| `avoid_Mt`, `avoidA_w` | Methane emission avoided by collection, Mt CO2e/yr and kg CO2e per kg of Route A H2 |
| `co2_stored_Mt` | CO2 stored in the capture variant, Mt/yr |
| `Pel_MW`, `B_feasible_sites` | Electrolysis capacity of Route B, MW, and number of sites where it can be built |
| `H2A_incr_kt`, `H2A_div_kt`, `H2A_lic_div_kt`, `gen_div_Mm3`, `H2B_comp_kt` | Results of the incremental, diversion and compression variants |
| `A_below_2` to `A_below_5`, `B_below_2` to `B_below_5`, `A_below_3_incr`, `A_below_3_div` | Share of the potential of each route with an LCOH below 2, 3, 4 and 5 USD/kg |
| `ciA_w`, `ciA_cc_w`, `ciA_cc_slip_w`, `ciB_w` | Production-weighted CI, kg CO2e/kg |
| `ciA_max` | Highest CI of Route A among the sites that produce H2, kg CO2e/kg |
| `A_cheaper_share` | Share of the Route A potential, at the sites where both routes can be built, at which Route A has the lower LCOH |
| `lcohA_wmed`, `lcohB_wmed`, `lcohA_cc_wmed`, `lcohA_incr_wmed`, `lcohA_div_wmed`, `lcohB_comp_wmed` | Production-weighted median LCOH, USD/kg |
| `H2A_licensed_kt`, `H2A_expired_kt`, `H2A_notreported_kt` and their `_lic_` counterparts | Route A potential of the sites with a current, expired or unreported license, base case and licensed-only variant |
| `st_*_kt`, `st_lic_*_kt` | Route A potential by remaining licensed life (more than 15, 10 to 15, 5 to 10, 2 to 5 and less than 2 years, expired, not reported) and by closure period |

## results/mc_params.csv

One row per realization with the values drawn for the global parameters, as quantiles (`u_`) or as values: collection efficiency quantile `u_eta`, methane fraction `F`, reformer efficiency quantile `u_E`, H2 formed relative to the stoichiometric maximum `eta_ref`, upgrading loss `s_up`, slip of engines `s_eng` and microturbines `s_mtur`, electrolyzer consumption `sec` (kWh/kg), reformer exponent `nR` and level `z_cREF` (standard normal), electrolyzer CAPEX `cEL` (USD/kW at 1 MW) and exponent `nE`, LFGcost-Web cost multiplier `fL`, discount rate `i`, capture cost `ccc`, transport and storage cost `tns` (USD/t), solvent regeneration heat `q_regen`, capture electricity `e_cc`, electricity price `pe` (USD/kWh), auxiliary electricity `e_aux` (kWh per m3 of LFG), upgrading exponent below range `n_up`, burner slip `s_burn`, PSA recovery quantile `u_psa`, emission of CO2 transport and storage `co2_ts` (kg CO2e per kg CO2), and decay constant quantiles `u_k1` to `u_k4` and methane capacity multipliers `l0m1` to `l0m4` of the four decay categories.

## results/variants_national.csv

One row per realization, with the same realizations as `mc_national.csv`. Each column is `VARIANT_METRIC`.

| Variant | Meaning |
|---|---|
| `base` | Base case, identical to `mc_national.csv` |
| `gen78` | Generation 22 % lower at every site |
| `fmin`, `fmin30` | Minimum load of 50 % for the reformer and the generator, or of 30 % and 25 % |
| `p75`, `max` | Design flow at the 75th percentile or at the maximum of the annual flows |
| `mod2` | Two identical trains, each sized at half of the maximum flow |
| `dyn` | Collection efficiency of each deposit by cover stage |
| `etaind`, `etarho` | Collection efficiency independent between sites, or correlated at 0.5 |
| `compdept` | Composition error common to the sites of each department |
| `capexpred`, `capexmix` | Reformer CAPEX from the predictive distribution of the fit or from the mixed model |
| `opp` | Recovered methane priced at its netback value as electricity, both routes |
| `orng15`, `orng20`, `eaux15` | OPEX of the cleanup and upgrading train multiplied by 1.5 or 2, or its electricity by 1.5 |
| `cc85`, `cc95`, `ccnots`, `ccleak1`, `ccleak10` | Capture fraction of 0.85 or 0.95, capture without transport and storage emissions, and 1 % or 10 % of the stored CO2 lost |
| `techfix` | Route B with the generator fixed at the modal technology of each site |

| Metric | Meaning |
|---|---|
| `H2A_kt`, `H2Alic_kt`, `H2B_kt` | National H2 of Route A in the base case and in the licensed-only variant, and of Route B, kt/yr |
| `lcohA_wmed`, `lcohAlic_wmed`, `lcohB_wmed`, `lcohAcc_wmed` | Production-weighted median LCOH, USD/kg |
| `lcohA_port`, `lcohB_port` | Aggregate LCOH of the portfolio, Equation 46 of the article, USD/kg |
| `lcohA_port30`, `H2A_top30_kt` | Aggregate LCOH and H2 of the 30 largest sites of the realization |
| `A_below_3` | Share of the Route A potential with an LCOH below 3 USD/kg |
| `A_cheaper_share` | As in `mc_national.csv` |
| `ciA_w`, `ciB_w`, `ciAcc_w` | Production-weighted CI, kg CO2e/kg |
| `co2_Mt` | CO2 stored, Mt/yr |
| `base_eta_w`, `dyn_eta_eff_w`, `dyn_eta_eff_lic_w` | Collection efficiency of the active sites over the window, weighted by generation, base case and dynamic variant without and with licensed deposits only |
| `dyn_eta_daily`, `dyn_eta_inter`, `dyn_eta_final` | Efficiencies of daily, intermediate and final cover drawn in the realization |
| `opp_netback_w`, `opp_share_pos` | Netback value of the methane weighted by the gas used, USD/m3, and share of the Route A potential at sites with a positive netback |
| `co2_ts`, `i` | Emission of CO2 transport and storage and discount rate drawn in the realization |

## results/variants_years.npz

Arrays of shape (10,000, 15), one row per realization and one column per year of `years` (2026 to 2040). `base`, `lic`, `dyn` and `dyn_lic` hold the national H2 of Route A, kg/yr, in the base case, in the licensed-only variant and in both with the collection efficiency by cover stage; `rec`, `rec_lic`, `rec_dyn` and `rec_dyn_lic` the recovered methane, m3/yr; `used` and `used_lic` the methane used by the plants, m3/yr.

## results/consistency_sites.csv and consistency_dj_annual.csv

One row per record of recovered methane (`consistency_sites.csv`) or per year at Doña Juana (`consistency_dj_annual.csv`). `rec_t_yr` is the recovered methane, annualized, t/yr, with `rec_t_yr_lo` and `rec_t_yr_hi` its range (for Doña Juana, at 50 % and 55 % methane); `gen_t_p10` to `gen_t_p90` the modeled generation of the same period, t/yr; `eta_p10` to `eta_p90` the implied collection efficiency, recovered over modeled generation per realization; `share_eta_gt1` the share of realizations in which the record exceeds the modeled generation; and `lfg_m3_h`, `ch4_pct`, `wells` and `tpd` the flare flow, methane content, number of wells and intake reported to the SUI.

## results/profiles.csv

Annual operation of Route A at Doña Juana, La Pradera and Colomba-El Guabal, at the default parameter values. `case` is `base`, `lic`, `dyn` or `dyn_lic`, as in `variants_years.npz`; `fmin` the minimum load (0 or 0.5); `gen_Mm3`, `rec_Mm3`, `used_Mm3` and `flared_Mm3` the methane generated, recovered, used and flared, Mm3/yr; `h2_kt` the H2 produced, kt/yr; `util` the methane used over the design flow; `design_Mm3` the design flow, Mm3/yr; `lcoh` the LCOH of the plant, USD/kg; and `closure` the year of the last licensed deposit. Table S9 of the Supplementary Materials lists the rows with `fmin` = 0.

## Other result files

| File | Content |
|---|---|
| `results/balances.csv` | Mass and energy balance of Route A per kg of H2 at Doña Juana and at Parque Ambiental Bioger, at the default values (Table 11 of the article) |
| `results/reformer_balance.json`, `reformer_balance_cases.csv` | Thermodynamic balance of the reformer island at 1.5 and 50 t/d and its one-at-a-time cases (Tables S4 and S5) |
| `results/capex_checks.json` | Prediction intervals of the reformer cost fit and its alternative specifications (Tables S6 and S7) |
| `results/consistency_gen_draws.npz` | Modeled methane generation, m3/yr, of the nine sites with a record of recovered methane (`sites`) in every year from 2005 to 2025 (`years`), for each realization (`G`, shape 10,000 × 9 × 21) |
| `results/sensitivity.csv` | Partial (`prcc`) and standardized (`srrc`) rank regression coefficients of each national output against each global parameter, with the coefficient of determination of the rank regression |
| `results/national_results.json`, `numbers_results.csv`, `numbers_defaults.csv`, `numbers_capex.csv`, `numbers_precip.csv` | The national statistics and the named values quoted in the text (columns `name` and `value`) |
| `results/validation.json`, `model_meta.json` | Checks against independent evidence and the settings of the run |
| `results/site_catalog.csv`, `site_crosswalk.csv`, `site_rows_assigned.csv`, `site_status_summary.csv` | The site catalog, the mapping of every registry row to a site and the status summary |
| `results/waste_site_year.csv`, `waste_municipality_site_year.csv`, `waste_pools.csv`, `waste_checks.json` | Deposits by site and year, observed, back-cast and projected, t, with the part beyond the licensed date |
| `results/composition_municipality.csv`, `composition_site.csv`, `composition_stats.json` | MSW composition by served municipality and by site and year |
| `results/site_precipitation.csv`, `precip_stats.json` | Precipitation and decay zone of each site |
| `results/population_municipal.csv` | Population by municipality and year, 2005 to 2042 |
| `results/capex_raw.csv`, `capex_normalised.csv`, `capex_final.json` | The reformer cost record, normalized to 2023 USD, and its fit |
| `results/mun.gpkg`, `dept.gpkg`, `mun_points.csv` | Simplified municipal and departmental boundaries and the representative point of each municipality |
