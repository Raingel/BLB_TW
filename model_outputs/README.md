# Final trained-model outputs

The model output directory contains 250 CNN-BiGRU-attention models:

```text
5 weather sources x 10 split seeds x 5 held-out test folds
```

Each fold directory contains:

- `best_model.pt`: PyTorch state dictionary selected by minimum validation loss.
- `split_indices.json`: train, validation, and held-out test record indices.
- `zscore.json`: feature-wise mean and standard deviation estimated only from that fold's training partition.
- `training_log.csv`: epoch-level training and validation history.
- `val_predictions.csv` and `test_predictions.csv`: fold predictions.
- `metrics.json`: fold performance summary.

Top-level files contain aggregated out-of-fold results. `cross_source_eval/` contains the fixed-model weather-source substitution evaluation.

The record indices correspond to the row order in both public training CSV files.

