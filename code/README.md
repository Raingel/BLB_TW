# Code notes

## Primary reproducible workflow

The scripts in `model/` are configured for the released privacy-safe sequence table:

```bash
python code/model/run_all_sources_cv_clean_0505.py
python code/model/evaluate_cross_source_clean_0505.py
```

The first command writes a new `reproduced_model_outputs/` directory. The original final artifacts are retained separately in `model_outputs/`.

## Public analysis code

`analysis/public_dataset_qc.py` reproduces basic dataset, sequence-completeness, and duplicate-signature checks directly from the privacy-safe public training table.

## Weather extraction

Weather-retrieval and field-curation scripts are not distributed because they require private field locations. The final extracted weather inputs needed to reproduce model training and cross-source evaluation are included in `data/`.
