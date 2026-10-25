# Provenance of the inputs

Every file of `inputs/` is either a public record redistributed without changes, a public record from which columns were removed, or a table prepared by the authors from public records. The public records remain subject to the terms of their publishers. The tables prepared by the authors are released under the Creative Commons Attribution 4.0 International license (see `LICENSE` in this folder), while the code of the repository is released under the MIT License.

## Public records

| File | Publisher and title | Source | Accessed | Changes |
|---|---|---|---|---|
| `population_2005-2042.xlsx` | National Administrative Department of Statistics (DANE), *Proyecciones de Población Municipal por Área 2005–2042* | https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion | 10 September 2026 | None |
| `DIVIPOLA_-_Códigos_cabeceras_-_Centros_poblados_20260718.xlsx` | DANE, *División Político-Administrativa de Colombia (DIVIPOLA)*, codes and coordinates of municipal seats and populated centers | Open data portal of Colombia, https://www.datos.gov.co | 18 July 2026 | None |
| `ideam_normales_nsz2-kzcq.csv` | Institute of Hydrology, Meteorology and Environmental Studies (IDEAM), *Normales Climatológicas de Colombia* | https://www.datos.gov.co/d/nsz2-kzcq | 12 September 2026 | None, cached copy of the whole dataset |
| `municipal_boundary_polygon.*` | Agustín Codazzi Geographic Institute (IGAC), *Límites de Entidades Territoriales de Colombia* (December 2020) | https://www.colombiaenmapas.gov.co/ | 12 January 2026 | None |
| `records/dona_juana_biogas_year.csv` | CGR Doña Juana S.A. E.S.P., *Estadística de Biogás y Plan de Manejo Integral de Residuos Sólidos* | https://www.datos.gov.co/dataset/Estadistica-Biogas-y-Plan-de-manejo-Integral-de-Re/qudk-p67d/about_data | 21 September 2026 | Annual means of the monthly records (flow in normal cubic meters per hour, electricity and reported emission reductions) |
| `records/sspd_disposal_national_2015_2024.csv` | Superintendency of Residential Public Services (SSPD), *Informe Nacional de Disposición Final de Residuos Sólidos en Colombia* (annual reports) | https://www.superservicios.gov.co | 17 September 2026 | National disposal by year, 2015 to 2024, as published by the SSPD |
| `records/sspd_site_inventory.csv` | SSPD, *Sistema Único de Información* (SUI), final-disposal module | https://www.sui.gov.co/web/aseo | 15 September 2026 | One row per registered site with its licensed end of life; only the columns read by the software are kept |
| `records/sui_landfill_sites.csv` | SSPD, SUI, technical register of the disposal sites | https://www.sui.gov.co/web/aseo | 15 September 2026 | Only the identification, status, opening date and licensed end of life are kept; contact details of the operators were removed |
| `records/sui_gas_systems.csv` | SSPD, SUI, gas systems reported by the operators | https://www.sui.gov.co/web/aseo | 15 September 2026 | Only the identification, intake, wells and flare flows are kept, matched to the site catalog |
| `composition/upme_dept_composition.csv` | Mining and Energy Planning Unit (UPME) and INERCO, *Valorización Energética de Residuos: Proyecto WTE Colombia*, Table 7 | https://bdigital.upme.gov.co/handle/001/1339 | 28 September 2026 | Transcribed |

The records of `records/` were compiled by the authors from the sources above, and they are published here in the form that the code reads.

## Tables prepared by the authors

| File | Content | Basis |
|---|---|---|
| `landfilling_site_code_imputed.csv` | The final-disposal registry of the SSPD for 2010 to 2024, one row per served municipality, receiving site and year, with the municipal code of the receiving site imputed where the registry did not report it | SUI and the annual reports of the SSPD; the column `imputed` marks the imputed codes and `match_method` and `match_confidence` the rule used |
| `site_overrides.csv` | Rules that merge the name variants of the large regional sites into canonical sites | Registry names, SSPD inspection reports and the documents of the Clean Development Mechanism (CDM) projects |
| `site_row_fixes.csv` | Registry rows reassigned to another site, with the reason | Registry records inconsistent with the other years |
| `site_facts.csv` | Opening years and existing methane destruction of the sites with documentary evidence | CDM project documents, SSPD inspection reports and the operator records of Doña Juana; the source of each value is in its row |
| `composition/msw_composition_measured.csv` | MSW composition measured in 56 municipalities, in percent by wet mass, one row per municipality; a blank cell is a fraction that the plan does not report, and `rubber_leather_bone` is the rubber, leather and bone that a plan reports within its residual | The integrated solid waste management plans (PGIRS) of the municipalities, listed with their period and sampling point in Table S3 of the Supplementary Materials |
