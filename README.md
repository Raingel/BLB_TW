# BLB_TW: Weather-source heterogeneity in rice bacterial leaf blight forecasting

This repository contains the weather inputs, final training datasets, analysis code, trained-model outputs, and numerical results supporting our study of weather-source heterogeneity in rice bacterial leaf blight (BLB) forecasting in Taiwan.

## Final dataset

The final dataset contains 404 labeled field-survey records:

- 202 complete event-control pairs
- 202 class-0 records and 202 class-1 records
- 198 anonymous field groups
- Surveys conducted from 2018 to 2025
- A 19-day survey-associated weather data (SAWD) window spanning days -21 to -3 before each survey
- Ten daily weather variables from five input pipelines: CWA, JRA3Q, MERRA2, ERA5-Land, and ERA5_2xCoarse

Exact field identifiers, survey dates, and coordinates are not released. `FieldID` is an anonymous grouping variable.

## Repository contents

### `data/`

- `blb_training_sawd_sequences_public.csv`: model-ready table. Each SAWD variable is stored as one 19-value array column.
- `blb_training_sawd_wide_public.csv`: fully flattened wide table. Each record has 950 daily SAWD feature columns (5 sources x 10 variables x 19 days).
- `data_dictionary.csv`: definitions, units, sources, and time positions for the flattened wide table.

### `code/`

- `model/`: CNN-BiGRU-attention cross-validation and cross-source evaluation scripts.
- `analysis/`: public-data QC and summary analysis.

### `model_outputs/`

Final artifacts from 10 split seeds x 5 folds x 5 weather sources (250 trained models). Each fold includes trained weights, split indices, train-only z-score parameters, training logs, predictions, and metrics. Aggregated within-source and cross-source outputs are also included.

### `results/`

Final compact statistical tables and manuscript-level numerical summaries.

## Reproducing model evaluation

Install the packages in `requirements.txt`, then run from the repository root:

```bash
python code/model/run_all_sources_cv_clean_0505.py
python code/model/evaluate_cross_source_clean_0505.py
```

The released training script is configured to use `data/blb_training_sawd_sequences_public.csv`, which preserves anonymous `FieldID` groups for grouped cross-validation.

## Data structure

Class labels:

- `0`: no observed bacterial leaf blight
- `1`: onset-related bacterial leaf blight record

Flattened SAWD columns follow:

```text
{SOURCE}_{VARIABLE}_d{DAY_BEFORE_SURVEY}
```

For example, `ERA5_temperature_2m_mean_d21` is ERA5-Land mean 2-m temperature 21 days before the survey.

## Integrity checks

`RELEASE_QC.json` summarizes the final release checks. `MANIFEST_SHA256.csv` lists file sizes and SHA-256 checksums.

## Citation and license

Please cite the associated manuscript when using these data or outputs.

Data in this repository are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Data privacy

To protect participating farms, the repository excludes original field names, survey dates, coordinates, survey notes, and station identifiers. `FieldID`, `pair_id`, and `record_id` are anonymous identifiers. These removals do not affect reproduction of the reported model training and cross-source evaluation.
