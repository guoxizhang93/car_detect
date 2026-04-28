#!/usr/bin/env python3
"""Deploy-side car stuck/fall-off risk prediction from CSV input."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gp_car_stuck_model_lib import (
    TemporalRiskConfig,
    apply_temporal_risk_filter,
    load_model,
    predict_with_values,
    print_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict car stuck/fall-off risk from CSV input")
    parser.add_argument("--model", required=True, type=Path, help="model file path")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV input path. Columns should match model features.",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="show required model feature order and exit",
    )
    parser.add_argument(
        "--temporal-filter",
        action="store_true",
        help="apply temporal risk filtering across CSV rows",
    )
    parser.add_argument("--window-size", type=int, default=10, help="temporal window size; 10 means about 1s at 100ms sampling")
    parser.add_argument("--high-threshold", type=float, default=0.7, help="instant high-risk threshold")
    parser.add_argument("--warning-count", type=int, default=4, help="high-risk count needed for warning")
    parser.add_argument("--critical-count", type=int, default=7, help="high-risk count needed for critical")
    parser.add_argument("--critical-avg-threshold", type=float, default=0.75, help="average risk needed for critical")
    parser.add_argument("--max-review-required", type=int, default=3, help="max review_required samples allowed for critical")
    parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA smoothing factor")
    parser.add_argument("--enter-threshold", type=float, default=0.7, help="hysteresis enter threshold")
    parser.add_argument("--exit-threshold", type=float, default=0.45, help="hysteresis exit threshold")
    return parser.parse_args()


def build_temporal_config(args):
    return TemporalRiskConfig(
        window_size=args.window_size,
        high_threshold=args.high_threshold,
        warning_count=args.warning_count,
        critical_count=args.critical_count,
        critical_avg_threshold=args.critical_avg_threshold,
        max_review_required=args.max_review_required,
        ema_alpha=args.ema_alpha,
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
    )


def read_csv_feature_rows(csv_path: Path, features: tuple[str, ...]) -> list[list[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"input CSV not found: {csv_path}")

    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("input CSV is missing header")

        missing = [feature for feature in features if feature not in reader.fieldnames]
        if missing:
            raise ValueError(f"input CSV missing feature columns: {', '.join(missing)}")

        for index, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[feature]) for feature in features])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {index} contains non-numeric feature values: {row}") from exc

    if not rows:
        raise ValueError("input CSV has no data rows")
    return rows


def main():
    args = parse_args()
    bundle = load_model(args.model)

    if args.show_features:
        print_json({"features": list(bundle.features)})
        return

    if args.input_csv is None:
        raise ValueError("prediction requires --input-csv")

    feature_rows = read_csv_feature_rows(args.input_csv, bundle.features)
    predictions = [predict_with_values(bundle, values) for values in feature_rows]

    if args.temporal_filter:
        predictions = apply_temporal_risk_filter(predictions, build_temporal_config(args))

    if len(predictions) == 1:
        print_json(predictions[0])
    else:
        print_json(predictions)


if __name__ == "__main__":
    main()
