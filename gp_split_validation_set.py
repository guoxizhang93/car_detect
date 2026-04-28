#!/usr/bin/env python3
"""Split a labeled CSV into stratified train and validation CSV files."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from gp_car_stuck_model_lib import DEFAULT_LABEL


def parse_args():
    parser = argparse.ArgumentParser(description="Split labeled CSV into train/validation sets")
    parser.add_argument("--csv", required=True, type=Path, help="input labeled CSV")
    parser.add_argument("--train-out", required=True, type=Path, help="output training CSV")
    parser.add_argument("--valid-out", required=True, type=Path, help="output validation CSV")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="label column name")
    parser.add_argument("--valid-size", type=float, default=0.2, help="validation ratio, default 0.2")
    parser.add_argument("--random-state", type=int, default=42, help="random seed")
    return parser.parse_args()


def read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"input CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("input CSV is missing header")
        return list(reader.fieldnames), list(reader)


def write_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: list[dict[str, str]], label_name: str, valid_size: float, random_state: int):
    if not 0.0 < valid_size < 1.0:
        raise ValueError("valid-size must be between 0 and 1")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        if label_name not in row:
            raise KeyError(f"CSV missing label column: {label_name}")
        label = row[label_name]
        if label not in {"0", "1", "0.0", "1.0"}:
            raise ValueError(f"row {index} label must be 0 or 1: {label}")
        grouped[str(int(float(label)))].append(row)

    if set(grouped) != {"0", "1"}:
        raise ValueError("input CSV must contain both label classes 0 and 1")

    rng = random.Random(random_state)
    train_rows: list[dict[str, str]] = []
    valid_rows: list[dict[str, str]] = []
    for label, label_rows in grouped.items():
        rows_copy = list(label_rows)
        rng.shuffle(rows_copy)
        valid_count = max(1, int(round(len(rows_copy) * valid_size)))
        if valid_count >= len(rows_copy):
            raise ValueError(f"valid-size leaves no training samples for label {label}")
        valid_rows.extend(rows_copy[:valid_count])
        train_rows.extend(rows_copy[valid_count:])

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def count_by_label(rows: list[dict[str, str]], label_name: str) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[str(int(float(row[label_name])))] += 1
    return counts


def main():
    args = parse_args()
    fieldnames, rows = read_rows(args.csv)
    train_rows, valid_rows = stratified_split(rows, args.label, args.valid_size, args.random_state)
    write_rows(args.train_out, fieldnames, train_rows)
    write_rows(args.valid_out, fieldnames, valid_rows)

    print(f"train_out={args.train_out}")
    print(f"valid_out={args.valid_out}")
    print(f"train_samples={len(train_rows)} labels={count_by_label(train_rows, args.label)}")
    print(f"valid_samples={len(valid_rows)} labels={count_by_label(valid_rows, args.label)}")


if __name__ == "__main__":
    main()
