#!/usr/bin/env python3
"""Select decision and temporal thresholds on a labeled validation CSV."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from gp_car_stuck_model_lib import (
    DEFAULT_LABEL,
    TemporalRiskConfig,
    TemporalRiskFilter,
    load_model,
    predict_with_values,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scan thresholds on labeled validation data")
    parser.add_argument("--model", required=True, type=Path, help="trained model file")
    parser.add_argument("--valid-csv", required=True, type=Path, help="labeled validation CSV")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="label column name")
    parser.add_argument("--decision-min", type=float, default=0.1)
    parser.add_argument("--decision-max", type=float, default=0.9)
    parser.add_argument("--decision-step", type=float, default=0.05)
    parser.add_argument("--high-min", type=float, default=0.5)
    parser.add_argument("--high-max", type=float, default=0.95)
    parser.add_argument("--high-step", type=float, default=0.05)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--warning-count", type=int, default=4)
    parser.add_argument("--critical-count", type=int, default=7)
    parser.add_argument("--critical-avg-threshold", type=float, default=0.75)
    parser.add_argument("--max-review-required", type=int, default=3)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--enter-threshold", type=float, default=0.7)
    parser.add_argument("--exit-threshold", type=float, default=0.45)
    parser.add_argument(
        "--false-negative-cost",
        type=float,
        default=5.0,
        help="business cost weight for missed risk samples",
    )
    parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=1.0,
        help="business cost weight for false alarms",
    )
    parser.add_argument("--metrics-out", type=Path, default=None, help="optional CSV path for all scanned metrics")
    parser.add_argument("--top-k", type=int, default=10, help="number of best rows to print")
    return parser.parse_args()


def float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be greater than 0")
    values = []
    value = start
    while value <= stop + step * 0.5:
        values.append(round(value, 10))
        value += step
    return values


def read_labeled_rows(csv_path: Path, features: tuple[str, ...], label_name: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"validation CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("validation CSV is missing header")
        missing = [name for name in [*features, label_name] if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"validation CSV missing columns: {', '.join(missing)}")

        for index, row in enumerate(reader, start=2):
            try:
                values = [float(row[feature]) for feature in features]
                label = int(float(row[label_name]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {index} contains invalid numeric values: {row}") from exc
            if label not in {0, 1}:
                raise ValueError(f"row {index} label must be 0 or 1: {label}")
            rows.append((values, label))

    if not rows:
        raise ValueError("validation CSV has no data rows")
    return rows


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    pred = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 0)

    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def temporal_metrics(
    labels: list[int],
    predictions: list[dict],
    high_threshold: float,
    args,
) -> dict:
    config = TemporalRiskConfig(
        window_size=args.window_size,
        high_threshold=high_threshold,
        warning_count=args.warning_count,
        critical_count=args.critical_count,
        critical_avg_threshold=args.critical_avg_threshold,
        max_review_required=args.max_review_required,
        ema_alpha=args.ema_alpha,
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
    )
    risk_filter = TemporalRiskFilter(config)
    states = [risk_filter.update(prediction)["state"] for prediction in predictions]
    alarm_pred = [1 if state in {"warning", "critical"} else 0 for state in states]

    tp = sum(1 for y, p in zip(labels, alarm_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, alarm_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, alarm_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, alarm_pred) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "temporal_tp": tp,
        "temporal_tn": tn,
        "temporal_fp": fp,
        "temporal_fn": fn,
        "temporal_precision": precision,
        "temporal_recall": recall,
        "temporal_f1": f1,
        "normal_count": sum(state == "normal" for state in states),
        "watch_count": sum(state == "watch" for state in states),
        "warning_count": sum(state == "warning" for state in states),
        "critical_count": sum(state == "critical" for state in states),
    }


def write_metrics(csv_path: Path, rows: list[dict]):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    bundle = load_model(args.model)
    labeled_rows = read_labeled_rows(args.valid_csv, bundle.features, args.label)
    labels = [label for _values, label in labeled_rows]
    predictions = [predict_with_values(bundle, values) for values, _label in labeled_rows]
    scores = [float(item["mean_prediction"]) for item in predictions]

    decision_values = float_range(args.decision_min, args.decision_max, args.decision_step)
    high_values = float_range(args.high_min, args.high_max, args.high_step)

    rows: list[dict] = []
    for decision_threshold in decision_values:
        decision_metrics = binary_metrics(labels, scores, decision_threshold)
        decision_cost = (
            args.false_negative_cost * decision_metrics["fn"]
            + args.false_positive_cost * decision_metrics["fp"]
        )
        for high_threshold in high_values:
            temporal = temporal_metrics(labels, predictions, high_threshold, args)
            temporal_cost = (
                args.false_negative_cost * temporal["temporal_fn"]
                + args.false_positive_cost * temporal["temporal_fp"]
            )
            rows.append(
                {
                    "decision_threshold": decision_threshold,
                    "high_threshold": high_threshold,
                    **decision_metrics,
                    "decision_cost": decision_cost,
                    **temporal,
                    "temporal_cost": temporal_cost,
                }
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["decision_cost"],
            -row["recall"],
            row["false_positive_rate"],
            row["temporal_cost"],
            -row["temporal_recall"],
        ),
    )

    result = {
        "validation_samples": len(labels),
        "label_counts": {"0": labels.count(0), "1": labels.count(1)},
        "ranking_rule": "min decision_cost, max recall, min false_positive_rate, min temporal_cost, max temporal_recall",
        "false_negative_cost": args.false_negative_cost,
        "false_positive_cost": args.false_positive_cost,
        "best": ranked[0],
        "top": ranked[: args.top_k],
        "temporal_config_base": {
            **asdict(
                TemporalRiskConfig(
                    window_size=args.window_size,
                    warning_count=args.warning_count,
                    critical_count=args.critical_count,
                    critical_avg_threshold=args.critical_avg_threshold,
                    max_review_required=args.max_review_required,
                    ema_alpha=args.ema_alpha,
                    enter_threshold=args.enter_threshold,
                    exit_threshold=args.exit_threshold,
                )
            ),
            "high_threshold": "scanned",
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.metrics_out is not None:
        write_metrics(args.metrics_out, rows)
        print(f"metrics_out={args.metrics_out}")


if __name__ == "__main__":
    main()
