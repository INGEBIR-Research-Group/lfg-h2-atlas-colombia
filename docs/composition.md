# Composition of the municipal solid waste

`scripts/04_build_composition.py` assigns a composition and a potential methane capacity (L0) to every municipality that delivers waste to a disposal site, and blends them into the composition of the mixture that each site receives in each year.

## Inputs

| File | Content |
|---|---|
| `inputs/composition/msw_composition_measured.csv` | Compositions measured in 56 municipalities and reported in their integrated solid waste management plans (PGIRS), in percent by wet mass. The plans are listed in Table S3 of the Supplementary Materials |
| `inputs/composition/upme_dept_composition.csv` | Departmental composition of Table 7 of the waste-to-energy study of the Mining and Energy Planning Unit (UPME, 2018), in its 10 categories, percent by mass |
| `results/waste_municipality_site_year.csv` | Deliveries of each municipality to each canonical site, written by `scripts/03_build_waste.py` |

## Method

Each municipality takes the first of two sources that is available.

1. **Measured characterization.** The composition reported by the plan of the municipality. Where a plan reports only the organic fraction and a residual lump, the lump is apportioned between the unreported degradable fractions and the inerts in the proportions of the departmental profile of that municipality, so that no measured fraction is lost and no gap is filled with zero.
2. **Departmental profile.** The UPME average of the department, mapped from its 10 categories onto the seven fractions of the model.
   - The organic category is split into 86 % food and 14 % garden waste (`FOOD_SHARE = 0.860`), since the study states that food accounts for 52.52 of the 61.06 percentage points of that category.
   - Category 10, nappies and other residue, is split into 70.5 % nappies and 29.5 % inerts (`NAPPY_SHARE_C10 = 0.055/0.078`), the ratio of the national nappy share of the 2016 policy characterization to the national share of that category. In the three departments that do not itemize their inorganic residue (Chocó, Nariño and Valle del Cauca), the nappy fraction is scaled with that residue.

The UPME study reports a profile for each of the 32 departments and for the Capital District, so no national average is needed, and the script stops with an error if a municipality has neither a measured nor a departmental profile.

The composition of a site is the tonnage-weighted blend of the municipalities that deliver to it, resolved by year, because the catchment of a site changes over time. L0 uses the four decay categories of the Colombia Landfill Gas Model, with food and 20 % of the nappies at 70, garden waste at 113, paper, textiles and 80 % of the nappies at 186, and wood and other slow material at 200 m3 CH4/Mg.

## Outputs

| File | Rows | Content |
|---|---|---|
| `results/composition_municipality.csv` | 1,104 | One row per served municipality, with the seven fractions, the source of the composition and L0 |
| `results/composition_site.csv` | 5,151 | One row per canonical site and year (the 1,868 sites of `results/site_catalog.csv`, 2010 to 2024), with the tonnage-weighted composition, L0, the number of contributing municipalities and the share of the intake backed by a measured characterization |
| `results/composition_stats.json` | | Summary figures used in the article |

## Summary of the results

- The measured characterizations cover 56 municipalities, while 1,048 take the departmental profile, but they cover 61.0 % of the tonnage disposed between 2010 and 2024, because those municipalities are the largest, and 64.0 % of the tonnage of the 248 sites active in 2024.
- The mass-weighted national L0 is 84.4 m3 CH4/Mg, and the carbon balance of the IPCC on the same composition gives 81.8 (DOC of 0.166), 3 % less.
- L0 ranges from 71.1 to 112.9 m3 CH4/Mg across the 1,868 sites of the catalog.
- Rubber, leather and bone, which 28 plans report within their residual lump (column `rubber_leather_bone` of the input), go to the slow category, as in the departmental profiles (`slow_moved` in `results/composition_stats.json`).
