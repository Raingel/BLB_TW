"""Reproduce basic QC summaries from the privacy-safe BLB training dataset."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "blb_training_sawd_sequences_public.csv"
OUT = ROOT / "reproduced_public_qc.json"

SOURCES = ["cwa", "jra3q", "merra2", "era5land", "era5_2xcoarse"]
VARIABLES = [
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


def source_matrix(df: pd.DataFrame, source: str) -> np.ndarray:
    columns = [f"{source}_sawd_{variable}" for variable in VARIABLES]
    records = []
    for _, row in df[columns].iterrows():
        arrays = [np.asarray(ast.literal_eval(row[column]), dtype=float) for column in columns]
        records.append(np.column_stack(arrays))
    return np.asarray(records)


def main() -> None:
    df = pd.read_csv(DATA)
    summary = {
        "records": int(len(df)),
        "pairs": int(df["pair_id"].nunique()),
        "anonymous_fields": int(df["FieldID"].nunique()),
        "class_counts": {str(k): int(v) for k, v in df["class"].value_counts().sort_index().items()},
        "missing_cells": int(df.isna().sum().sum()),
        "source_qc": {},
    }
    for source in SOURCES:
        matrix = source_matrix(df, source)
        signatures = pd.DataFrame(matrix.reshape(len(matrix), -1)).astype(str).agg("|".join, axis=1)
        collision_labels = df.assign(signature=signatures).groupby("signature")["class"].nunique()
        summary["source_qc"][source] = {
            "shape": list(matrix.shape),
            "finite": bool(np.isfinite(matrix).all()),
            "unique_signatures": int(signatures.nunique()),
            "mixed_label_signatures": int((collision_labels > 1).sum()),
        }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
