import copy
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parents[1] / "data" / "blb_training_sawd_sequences_public.csv"
OUT_DIR = SCRIPT_DIR.parents[1] / "reproduced_model_outputs"

SOURCES = {
    "CWA": "cwa_sawd",
    "JRA3Q": "jra3q_sawd",
    "MERRA2": "merra2_sawd",
    "ERA5Land": "era5land_sawd",
    "ERA5_2xCoarse": "era5_2xcoarse_sawd",
}

VARIABLE_BASES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_speed_10m_mean",
    "wind_speed_10m_min",
    "precipitation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
]

BATCH_SIZE = 16
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 40
K_FOLDS = 5
MAX_RETRIES = 10

ENSEMBLE_SEEDS = [6469, 174, 9919, 3299, 3489, 5168, 5319, 5018, 1497, 4065]
TRAIN_SEEDS = [2290, 4767, 7851, 9034, 9351, 4407, 2117, 1255, 3589, 3796]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeatherDataset(Dataset):
    def __init__(self, sequences, labels, mean_std=None):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.sequences = np.array(sequences, dtype=np.float32)

        if mean_std is None:
            reshaped = self.sequences.reshape(-1, len(VARIABLE_BASES))
            self.mean = reshaped.mean(axis=0)
            self.std = reshaped.std(axis=0) + 1e-8
        else:
            self.mean, self.std = mean_std

        self.sequences = (self.sequences - self.mean) / self.std
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class CNN_BiGRU_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=len(VARIABLE_BASES), out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(32, 32, batch_first=True, num_layers=2, bidirectional=True, dropout=0.3)
        self.attn_weights = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        gru_in = cnn_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)
        attn_scores = self.attn_weights(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)
        out = self.fc(context)
        return torch.sigmoid(out).squeeze()


def parse_sequences(df: pd.DataFrame, prefix: str) -> np.ndarray:
    seqs = []
    columns = [f"{prefix}_{base}" for base in VARIABLE_BASES]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {prefix}: {missing}")

    for row_idx, row in df.iterrows():
        seq = []
        for col in columns:
            values = json.loads(row[col])
            if len(values) != 19:
                raise ValueError(f"{col} row {row_idx} has length {len(values)}, expected 19")
            seq.append(values)
        arr = np.array(seq, dtype=np.float32).T
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite SAWD value in {prefix} row {row_idx}")
        seqs.append(arr)
    return np.stack(seqs, axis=0)


def make_splits(df: pd.DataFrame, split_seed: int, fold_idx: int):
    rs = np.random.RandomState(split_seed)
    unique_fields = df["FieldID"].astype(str).unique()
    rs.shuffle(unique_fields)
    folds = np.array_split(unique_fields, K_FOLDS)
    test_fields = folds[fold_idx]
    train_val_fields = np.concatenate([folds[j] for j in range(K_FOLDS) if j != fold_idx])

    rs_val = np.random.RandomState(split_seed + fold_idx)
    rs_val.shuffle(train_val_fields)
    val_size = max(1, int(len(unique_fields) * 0.15))
    val_fields = train_val_fields[:val_size]
    train_fields = train_val_fields[val_size:]

    train_idx = df[df["FieldID"].astype(str).isin(train_fields)].index.to_numpy(dtype=int)
    val_idx = df[df["FieldID"].astype(str).isin(val_fields)].index.to_numpy(dtype=int)
    test_idx = df[df["FieldID"].astype(str).isin(test_fields)].index.to_numpy(dtype=int)
    return train_idx, val_idx, test_idx


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            preds = model(seq)
            if len(preds.shape) == 0:
                preds = preds.unsqueeze(0)
            loss = criterion(preds, labels)
            total_loss += loss.item() * seq.size(0)
            n += seq.size(0)
    return total_loss / n


def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for seq, _ in loader:
            seq = seq.to(device)
            pred = model(seq)
            probs.extend(np.atleast_1d(pred.detach().cpu().numpy()).astype(float).tolist())
    return np.array(probs, dtype=float)


def metric_dict(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "class0": int((y_true == 0).sum()),
        "class1": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_one_fold(source_name, seqs, labels, train_idx, val_idx, test_idx, fold_dir, train_seed):
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ds = WeatherDataset(seqs[train_idx], labels[train_idx])
    val_ds = WeatherDataset(seqs[val_idx], labels[val_idx], mean_std=(train_ds.mean, train_ds.std))
    test_ds = WeatherDataset(seqs[test_idx], labels[test_idx], mean_std=(train_ds.mean, train_ds.std))

    (fold_dir / "split_indices.json").write_text(
        json.dumps({"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}, indent=2),
        encoding="utf-8",
    )
    (fold_dir / "zscore.json").write_text(
        json.dumps({"mean": train_ds.mean.tolist(), "std": train_ds.std.tolist()}, indent=2),
        encoding="utf-8",
    )

    criterion = nn.BCELoss()
    best_model_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    training_log = []
    used_train_seed = train_seed
    retry_count = 0

    for retry in range(MAX_RETRIES + 1):
        used_train_seed = train_seed + retry * 1000
        set_seed(used_train_seed)
        model = CNN_BiGRU_Attention().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator()
        generator.manual_seed(used_train_seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        current_best_val_loss = float("inf")
        current_best_epoch = -1
        current_best_state = None
        current_log = []

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            seen_n = 0
            for seq, batch_labels in train_loader:
                seq, batch_labels = seq.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                preds = model(seq)
                if len(preds.shape) == 0:
                    preds = preds.unsqueeze(0)
                loss = criterion(preds, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * seq.size(0)
                seen_n += seq.size(0)

            train_loss = train_loss / seen_n
            val_loss = evaluate_loss(model, val_loader, criterion)
            current_log.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_seen_n": seen_n,
                    "train_total_n": len(train_ds),
                    "train_seed": used_train_seed,
                    "retry": retry,
                }
            )

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                current_best_epoch = epoch
                current_best_state = copy.deepcopy(model.state_dict())

        if current_best_epoch >= 5 or retry == MAX_RETRIES:
            retry_count = retry
            best_epoch = current_best_epoch
            best_val_loss = current_best_val_loss
            best_model_state = current_best_state
            training_log = current_log
            break

    model = CNN_BiGRU_Attention().to(device)
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, fold_dir / "best_model.pt")
    pd.DataFrame(training_log).to_csv(fold_dir / "training_log.csv", index=False)

    val_prob = predict(model, val_ds)
    test_prob = predict(model, test_ds)
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    pd.DataFrame(
        {
            "row_index": val_idx,
            "class": val_labels.astype(int),
            "prob": val_prob,
            "pred": (val_prob >= 0.5).astype(int),
        }
    ).to_csv(fold_dir / "val_predictions.csv", index=False)
    pd.DataFrame(
        {
            "row_index": test_idx,
            "class": test_labels.astype(int),
            "prob": test_prob,
            "pred": (test_prob >= 0.5).astype(int),
        }
    ).to_csv(fold_dir / "test_predictions.csv", index=False)

    metrics = {
        "source": source_name,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "used_train_seed": int(used_train_seed),
        "retry_count": int(retry_count),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "val": metric_dict(val_labels, val_prob),
        "test": metric_dict(test_labels, test_prob),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def aggregate_outputs(df, out_dir):
    rows = []
    pred_rows = []
    for metrics_path in out_dir.glob("*/seed_*/fold_*/metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        rel = metrics_path.relative_to(out_dir)
        source = rel.parts[0]
        seed = int(rel.parts[1].replace("seed_", ""))
        fold = int(rel.parts[2].replace("fold_", ""))
        row = {
            "source": source,
            "seed": seed,
            "fold": fold,
            "best_epoch": metrics["best_epoch"],
            "best_val_loss": metrics["best_val_loss"],
            "used_train_seed": metrics["used_train_seed"],
            "retry_count": metrics["retry_count"],
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "n_test": metrics["n_test"],
        }
        for key, val in metrics["test"].items():
            row[f"test_{key}"] = val
        rows.append(row)

        pred = pd.read_csv(metrics_path.parent / "test_predictions.csv")
        pred["source"] = source
        pred["seed"] = seed
        pred["fold"] = fold
        pred_rows.append(pred)

    fold_metrics = pd.DataFrame(rows).sort_values(["source", "seed", "fold"])
    fold_metrics.to_csv(out_dir / "all_fold_metrics.csv", index=False)
    oof = pd.concat(pred_rows, ignore_index=True)
    oof.to_csv(out_dir / "all_oof_predictions.csv", index=False)

    summary_rows = []
    per_record_rows = []
    for source, g in oof.groupby("source"):
        pooled = metric_dict(g["class"].to_numpy(), g["prob"].to_numpy())
        rec = g.groupby("row_index", as_index=False).agg(class_true=("class", "first"), prob_mean=("prob", "mean"))
        per_record = metric_dict(rec["class_true"].to_numpy(), rec["prob_mean"].to_numpy())
        per_record_rows.append(rec.assign(source=source))
        for metric_name, values in fold_metrics[fold_metrics["source"] == source][
            ["test_accuracy", "test_mcc", "test_roc_auc"]
        ].items():
            summary_rows.append(
                {
                    "source": source,
                    "summary": f"fold_{metric_name}",
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        summary_rows.append({"source": source, "summary": "pooled_oof_mcc", "mean": pooled["mcc"]})
        summary_rows.append({"source": source, "summary": "pooled_oof_accuracy", "mean": pooled["accuracy"]})
        summary_rows.append({"source": source, "summary": "pooled_oof_roc_auc", "mean": pooled["roc_auc"]})
        summary_rows.append({"source": source, "summary": "per_record_mean_mcc", "mean": per_record["mcc"]})
        summary_rows.append({"source": source, "summary": "per_record_mean_accuracy", "mean": per_record["accuracy"]})
        summary_rows.append({"source": source, "summary": "per_record_mean_roc_auc", "mean": per_record["roc_auc"]})

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.concat(per_record_rows, ignore_index=True).to_csv(out_dir / "all_oof_predictions_per_record_mean.csv", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    if "FieldID" not in df.columns:
        df["FieldID"] = df["canonical_field_key"]
    labels = df["class"].astype(int).to_numpy()

    config = {
        "data_path": str(DATA_PATH),
        "out_dir": str(OUT_DIR),
        "sources": SOURCES,
        "variable_bases": VARIABLE_BASES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "k_folds": K_FOLDS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "device": str(device),
        "notes": "Clean 0505 metadata-deduplicated dataset. Model initialization and DataLoader shuffle are seeded with fold-specific train_seed.",
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Loading {DATA_PATH}", flush=True)
    print(f"Rows={len(df)} class_counts={df['class'].value_counts().sort_index().to_dict()} device={device}", flush=True)

    all_source_sequences = {source: parse_sequences(df, prefix) for source, prefix in SOURCES.items()}

    for source_i, (source_name, _) in enumerate(SOURCES.items()):
        print(f"\n=== Source {source_name} ({source_i + 1}/{len(SOURCES)}) ===", flush=True)
        seqs = all_source_sequences[source_name]
        for run_i, (split_seed, train_seed_base) in enumerate(zip(ENSEMBLE_SEEDS, TRAIN_SEEDS), start=1):
            print(f"--- Split seed {split_seed} ({run_i}/{len(ENSEMBLE_SEEDS)}) ---", flush=True)
            for fold_i in range(K_FOLDS):
                train_idx, val_idx, test_idx = make_splits(df, split_seed, fold_i)
                fold_train_seed = train_seed_base + source_i * 10000 + fold_i * 100
                fold_dir = OUT_DIR / source_name / f"seed_{split_seed}" / f"fold_{fold_i + 1}"
                completed_metrics = None
                if (fold_dir / "metrics.json").exists():
                    completed_metrics = json.loads((fold_dir / "metrics.json").read_text())
                if (
                    completed_metrics is not None
                    and completed_metrics.get("best_epoch", 0) >= 5
                    and (fold_dir / "best_model.pt").exists()
                    and (fold_dir / "test_predictions.csv").exists()
                ):
                    print(
                        f"{source_name} seed {split_seed} fold {fold_i + 1}: "
                        f"already complete, test_mcc={completed_metrics['test']['mcc']:.3f}",
                        flush=True,
                    )
                    continue
                metrics = train_one_fold(
                    source_name, seqs, labels, train_idx, val_idx, test_idx, fold_dir, fold_train_seed
                )
                print(
                    f"{source_name} seed {split_seed} fold {fold_i + 1}: "
                    f"epoch={metrics['best_epoch']} val_loss={metrics['best_val_loss']:.4f} "
                    f"test_mcc={metrics['test']['mcc']:.3f} auc={metrics['test']['roc_auc']:.3f}",
                    flush=True,
                )

    aggregate_outputs(df, OUT_DIR)
    print(f"\nDone. Artifacts saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
