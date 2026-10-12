# Antimicrobial resistance prediction

This project uses the ARMD dataset to predict whether a tested antibiotic result is susceptible or resistant. It compares logistic regression with LightGBM.

The prepared dataset has 182,487 rows from 15,952 patients and 28,498 culture events. We use a fixed 25% sample of patients to keep the project manageable.

## Start here

- [Figures](results/advanced/figures): extra analysis graphs in PNG format.
- [Figure notes](results/advanced/figure_index.json): what each graph shows and which tables it uses.
- [Main results](results/models/model_comparison.csv): model scores.
- [Extra results workbook](results/advanced/advanced_results.xlsx): tables behind the extra analysis.
- [Prepared dataset](results/analytical.parquet): the final data used by the models. A compressed CSV copy is also in `results/`.

The graphs keep axis names and tick labels. Explanations and source details are in the figure notes and results workbook. [Model curve colours and panel order](results/figure_notes.json) are listed separately from the images.

## What we predict

Each row describes one patient, one culture event, one organism and one candidate antibiotic.

- Susceptible is `1`.
- Resistant is `0`.
- Intermediate, unknown and conflicting results are excluded.
- Drugs without a recorded result are not given a label.

The nine antibiotics are listed in [config.json](config.json). This is our selected ARMD panel. The assignment mentions an instructor's list, but no separate list was supplied.

Inputs include the candidate drug, age group, gender code, culture type, ward information, selected lab and vital summaries, and earlier antibiotic and resistance records. Current organism identity is kept for grouping results, but is not a model input because it is usually unknown when the culture is ordered.

The [data dictionary](docs/final_data_dictionary.csv) explains each field. The [feature mapping](docs/feature_harmonization_matrix.xlsx) shows where the fields come from.

## How the data are kept separate

Patients are split into training, validation and test sets. A patient's rows stay in one set.

The code learns how to fill missing values and scale numbers from training data only. Missing values are kept as missing in the prepared dataset. Separate training patients are used to adjust the model's probabilities.

Validation data are used to choose the model and decision cutoff. Test data are used to check the final result. Class weights are compared during training. Validation and test data keep their original class balance.

## Main results

Weighted LightGBM was selected using validation results. Its test scores are:

| Measure | Score |
| --- | ---: |
| Average AUROC across drugs | 0.707 |
| Average precision for resistance, averaged across drugs | 0.381 |
| Average precision across both classes and drugs | 0.636 |
| Balanced accuracy | 0.730 |
| Resistant precision | 0.377 |
| Resistant recall | 0.732 |
| Brier score | 0.118 |

AUROC measures ranking. Resistant precision tells us how often a resistance prediction is right. Resistant recall tells us how much resistance is found. Brier score measures probability error, so lower is better. The [method notes](docs/methods.html) explain the measures in more detail.

The extra analysis checks which feature groups help, results for each drug, probability quality, missing data, and the effect of changing the cutoff. These are follow-up checks. They do not select a new model using test results.

## Files in this project

| Folder or file | What it contains |
| --- | --- |
| `src/armd/` | Data cleaning, features, models and evaluation |
| `scripts/` | Commands that run the work and create the outputs |
| `tests/` | Checks for the main data and model rules |
| `data/raw/` | Put the original ARMD files here |
| `docs/` | Field definitions, feature mapping and method notes |
| `results/` | Prepared data, scores, figures and saved models |
| `config.json` | Sample size, random seed and selected antibiotics |
| `requirements.txt` | Installs the package versions listed in `requirements-lock.txt` |

## Set up Python

This project was tested on Windows with Python 3.12.10. Open PowerShell in the project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You do not need to activate the environment. The commands below use its Python directly.

## Get the source data

Data source: [ARMD on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp).

Study paper: [ARMD in Scientific Data](https://www.nature.com/articles/s41597-025-05649-7).

The saved results use Dryad version `403223`, dated 22 October 2025. The raw CSV files are large, so they are kept out of Git. The file list and checksums are included in `data/raw/`.

If you have the supplied `Data_Intelligence.zip`, extract its source files with:

```powershell
.\.venv\Scripts\python.exe scripts\unpack_sources.py "$env:USERPROFILE\Downloads\Data_Intelligence.zip"
```

Or place the original CSV files directly in `data/raw/`. Keep their original names. The model uses these seven tables:

```text
microbiology_cultures_cohort.csv
microbiology_cultures_demographics.csv
microbiology_cultures_ward_info.csv
microbiology_cultures_labs.csv
microbiology_cultures_vitals.csv
microbiology_cultures_antibiotic_class_exposure.csv
microbiology_cultures_microbial_resistance.csv
```

Use all the supplied files to repeat the full source audit. `verified_downloads.json` records the files used for the saved results. The unpacking script checks the source files against the saved publisher checksums.

## Check the saved results

The prepared data, saved models and results are included in Git. You can check them without downloading the large raw CSV files:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\validate_package.py
.\.venv\Scripts\python.exe scripts\validate_advanced.py --require-present
```

The current version passes 32 tests, 48 checks of the main model scores, and 1,460 checks of the extra analysis.

## Run everything again

First add the raw source files. The commands below replace the generated results in this folder. The saved extra-analysis folder is moved to a dated backup first, because it protects the original model files from being changed during follow-up checks.

```powershell
if (Test-Path results\advanced) {
    $analysisBackup = ".cache\advanced_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
    New-Item -ItemType Directory -Force .cache | Out-Null
    Move-Item -LiteralPath results\advanced -Destination $analysisBackup
}

.\.venv\Scripts\python.exe scripts\make_documentation.py
.\.venv\Scripts\python.exe scripts\profile_source.py
.\.venv\Scripts\python.exe scripts\run_pipeline.py --raw-dir data/raw --output results --config config.json
.\.venv\Scripts\python.exe scripts\summarize_results.py
.\.venv\Scripts\python.exe scripts\run_advanced_analysis.py
```

The last command creates the figures and workbook, checks the results, and builds `submission/ARMD_project_package.zip`. The ZIP includes the prepared data and outputs, but not the large raw CSV files or reports. Keep the same source files, settings and package versions to repeat the saved analysis. Times and file checksums can change between runs.

To only rebuild the figures, workbook and ZIP from the saved analysis, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_advanced_analysis.py --presentation-only
```

Reports are optional and are kept out of Git. To create them locally and include them in the ZIP, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_advanced_analysis.py --presentation-only --include-reports
```

If you package existing outputs separately, reports are included only with `scripts/package_submission.py --include-reports`.

## Limits

- The supplied files do not identify the two hospitals separately. We cannot test performance at each hospital.
- Dates are shifted for each patient. They cannot show real changes over calendar years.
- Exact lab and vital result-release times are unavailable. Their time windows follow the source documentation.
- Results cover drugs with recorded, clear labels. They may not apply to drugs without recorded results.
- The extra analysis uses data already examined in the main study, so its findings need a separate check on new data.


