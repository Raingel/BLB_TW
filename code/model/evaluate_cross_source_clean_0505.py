import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

from run_all_sources_cv_clean_0505 import (
    BATCH_SIZE,
    CNN_BiGRU_Attention,
    DATA_PATH,
    OUT_DIR,
    SOURCES,
    VARIABLE_BASES,
    make_splits,
    parse_sequences,
)


CROSS_DIR = OUT_DIR / "cross_source_eval"
CROSS_DIR.mkdir(parents=True, exist_ok=True)


def metric_dict(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": auc,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_state(model_path, normalized_seq):
    model = CNN_BiGRU_Attention()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    probs = []
    x = torch.tensor(normalized_seq, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            pred = model(x[start : start + BATCH_SIZE])
            probs.extend(np.atleast_1d(pred.detach().cpu().numpy()).astype(float).tolist())
    return np.array(probs, dtype=float)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    labels = df["class"].astype(int).to_numpy()
    arrays = {source: parse_sequences(df, prefix) for source, prefix in SOURCES.items()}

    fold_metrics = pd.read_csv(OUT_DIR / "all_fold_metrics.csv")
    rows = []
    pred_rows = []
    threshold_rows = []

    for _, fold_row in fold_metrics.iterrows():
        train_source = fold_row["source"]
        seed = int(fold_row["seed"])
        fold = int(fold_row["fold"])
        fold_dir = OUT_DIR / train_source / f"seed_{seed}" / f"fold_{fold}"
        split = json.loads((fold_dir / "split_indices.json").read_text())
        train_idx = np.array(split["train"], dtype=int)
        test_idx = np.array(split["test"], dtype=int)
        y_test = labels[test_idx]
        z = json.loads((fold_dir / "zscore.json").read_text())
        train_mean = np.array(z["mean"], dtype=np.float32).reshape(1, 1, -1)
        train_std = np.array(z["std"], dtype=np.float32).reshape(1, 1, -1)
        model_path = fold_dir / "best_model.pt"

        for eval_source, eval_seq_all in arrays.items():
            eval_seq = eval_seq_all[test_idx]
            raw_norm = (eval_seq - train_mean) / train_std
            raw_prob = predict_state(model_path, raw_norm)
            md = metric_dict(y_test, raw_prob)
            rows.append(
                {
                    "train_source": train_source,
                    "eval_source": eval_source,
                    "seed": seed,
                    "fold": fold,
                    "mode": "raw_substitution",
                    **md,
                }
            )
            for idx, y, p in zip(test_idx, y_test, raw_prob):
                pred_rows.append(
                    {
                        "train_source": train_source,
                        "eval_source": eval_source,
                        "seed": seed,
                        "fold": fold,
                        "mode": "raw_substitution",
                        "row_index": int(idx),
                        "class": int(y),
                        "prob": float(p),
                    }
                )

            eval_train = eval_seq_all[train_idx]
            eval_mean = eval_train.reshape(-1, len(VARIABLE_BASES)).mean(axis=0).reshape(1, 1, -1)
            eval_std = (eval_train.reshape(-1, len(VARIABLE_BASES)).std(axis=0) + 1e-8).reshape(1, 1, -1)
            corrected_norm = (eval_seq - eval_mean) / eval_std
            corrected_prob = predict_state(model_path, corrected_norm)
            md = metric_dict(y_test, corrected_prob)
            rows.append(
                {
                    "train_source": train_source,
                    "eval_source": eval_source,
                    "seed": seed,
                    "fold": fold,
                    "mode": "mean_variance_corrected",
                    **md,
                }
            )
            for idx, y, p in zip(test_idx, y_test, corrected_prob):
                pred_rows.append(
                    {
                        "train_source": train_source,
                        "eval_source": eval_source,
                        "seed": seed,
                        "fold": fold,
                        "mode": "mean_variance_corrected",
                        "row_index": int(idx),
                        "class": int(y),
                        "prob": float(p),
                    }
                )

            for threshold in [0.4, 0.5, 0.6]:
                md = metric_dict(y_test, raw_prob, threshold=threshold)
                threshold_rows.append(
                    {
                        "train_source": train_source,
                        "eval_source": eval_source,
                        "seed": seed,
                        "fold": fold,
                        "mode": "raw_substitution",
                        "threshold": threshold,
                        **md,
                    }
                )

    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    thresholds = pd.DataFrame(threshold_rows)
    metrics.to_csv(CROSS_DIR / "cross_source_fold_metrics.csv", index=False)
    preds.to_csv(CROSS_DIR / "cross_source_predictions.csv", index=False)
    thresholds.to_csv(CROSS_DIR / "cross_source_threshold_metrics.csv", index=False)

    summary = (
        metrics.groupby(["mode", "train_source", "eval_source"])[["accuracy", "mcc", "roc_auc", "sensitivity", "specificity"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns]
    summary.to_csv(CROSS_DIR / "cross_source_metric_summary.csv", index=False)

    within = metrics[metrics["train_source"] == metrics["eval_source"]]
    within = within[["mode", "train_source", "seed", "fold", "mcc", "roc_auc"]].rename(
        columns={"mcc": "within_mcc", "roc_auc": "within_roc_auc"}
    )
    delta = metrics.merge(within, on=["mode", "train_source", "seed", "fold"], how="left")
    delta["delta_mcc_vs_within"] = delta["mcc"] - delta["within_mcc"]
    delta["delta_auc_vs_within"] = delta["roc_auc"] - delta["within_roc_auc"]
    delta.to_csv(CROSS_DIR / "cross_source_delta_vs_within.csv", index=False)
    delta_summary = (
        delta.groupby(["mode", "train_source", "eval_source"])[["delta_mcc_vs_within", "delta_auc_vs_within"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    delta_summary.columns = ["_".join([x for x in col if x]) for col in delta_summary.columns]
    delta_summary.to_csv(CROSS_DIR / "cross_source_delta_summary.csv", index=False)

    threshold_summary = (
        thresholds.groupby(["train_source", "eval_source", "threshold"])[["accuracy", "mcc", "sensitivity", "specificity"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    threshold_summary.columns = ["_".join([x for x in col if x]) for col in threshold_summary.columns]
    threshold_summary.to_csv(CROSS_DIR / "cross_source_threshold_summary.csv", index=False)

    print("Wrote cross-source metrics to", CROSS_DIR)
    print(delta_summary[delta_summary["mode"].eq("raw_substitution")].sort_values("delta_mcc_vs_within_mean").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
