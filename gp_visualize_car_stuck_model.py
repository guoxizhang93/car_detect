#!/usr/bin/env python3
"""Plot GP training and prediction results with matplotlib."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gp_car_stuck_model_lib import (
    DEFAULT_LABEL,
    TemporalRiskConfig,
    apply_temporal_risk_filter,
    build_dataset,
    load_model,
    predict_with_values,
    read_csv_rows,
)


STATE_COLORS = {
    "normal": "#e5e7eb",
    "watch": "#fde68a",
    "warning": "#fb923c",
    "critical": "#ef4444",
}
RELIABILITY_COLORS = {
    "reliable": "#16a34a",
    "caution": "#f59e0b",
    "review_required": "#7c2d12",
}


def lazy_import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少绘图依赖，请先安装: pip install matplotlib"
        ) from exc
    return np, plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GP car stuck/fall-off model results")
    parser.add_argument("--model", required=True, type=Path, help="model file path")
    parser.add_argument("--train-csv", required=True, type=Path, help="training CSV with labels")
    parser.add_argument("--valid-csv", type=Path, default=None, help="optional validation CSV with labels")
    parser.add_argument("--predict-csv", required=True, type=Path, help="prediction/replay CSV")
    parser.add_argument("--threshold-metrics-csv", type=Path, default=None, help="optional threshold scan metrics CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("gp_visualizations"), help="output folder")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="label column in training CSV")
    parser.add_argument("--feature-x", default="speed", help="feature used for x-axis")
    parser.add_argument("--feature-y", default="drive_current", help="feature used for y-axis")
    parser.add_argument("--grid-size", type=int, default=80, help="2D surface grid resolution")
    parser.add_argument("--window-size", type=int, default=10, help="temporal filter window size")
    parser.add_argument("--high-threshold", type=float, default=0.7, help="temporal high-risk threshold")
    parser.add_argument("--dpi", type=int, default=150, help="PNG output DPI")
    return parser.parse_args()


def read_prediction_feature_rows(csv_path: Path, features: tuple[str, ...]) -> list[list[float]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("prediction CSV is missing header")

        missing = [feature for feature in features if feature not in reader.fieldnames]
        if missing:
            raise ValueError(f"prediction CSV missing feature columns: {', '.join(missing)}")

        rows = []
        for index, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[feature]) for feature in features])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {index} contains non-numeric feature values: {row}") from exc

    if not rows:
        raise ValueError("prediction CSV has no data rows")
    return rows


def range_with_margin(values, ratio: float = 0.08) -> tuple[float, float]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        margin = max(abs(vmin) * ratio, 1.0)
    else:
        margin = (vmax - vmin) * ratio
    return vmin - margin, vmax + margin


def select_best_thresholds(metrics_rows: list[dict[str, float]]) -> dict[str, float]:
    return min(
        metrics_rows,
        key=lambda row: (
            row["decision_cost"],
            -row["recall"],
            row["false_positive_rate"],
            row["temporal_cost"],
            -row["temporal_recall"],
        ),
    )


def save_training_effect(bundle, x_data, y_data, feature_x: str, feature_y: str, decision_threshold: float, output_path: Path, dpi: int):
    np, plt = lazy_import_plotting()
    features = bundle.features
    ix = features.index(feature_x)
    iy = features.index(feature_y)

    pred_mean, _pred_std = bundle.pipeline.predict(x_data, return_std=True)
    pred_mean = np.clip(pred_mean, 0.0, 1.0)
    pred_label = (pred_mean >= decision_threshold).astype(int)
    accuracy = float((pred_label == y_data.astype(int)).mean())
    mae = float(np.abs(y_data - pred_mean).mean())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    fig.suptitle(
        f"Training effect: samples={len(x_data)}, in-sample accuracy={accuracy:.3f}, MAE={mae:.3f}, threshold={decision_threshold:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    normal = y_data == 0
    risk = y_data == 1
    ax.scatter(x_data[normal, ix], x_data[normal, iy], s=16, c="#2563eb", alpha=0.55, label="label 0 normal")
    ax.scatter(x_data[risk, ix], x_data[risk, iy], s=16, c="#dc2626", alpha=0.55, label="label 1 risk")
    ax.set_title("Training labels in feature space")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    bins = np.linspace(0.0, 1.0, 22)
    ax.hist(pred_mean[normal], bins=bins, alpha=0.70, color="#2563eb", label="label 0")
    ax.hist(pred_mean[risk], bins=bins, alpha=0.70, color="#dc2626", label="label 1")
    ax.axvline(decision_threshold, color="#111827", linestyle="--", linewidth=1.5, label=f"decision threshold={decision_threshold:.2f}")
    ax.set_title("Risk score separation on training data")
    ax.set_xlabel("mean_prediction")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_label_vs_prediction(bundle, x_data, y_data, decision_threshold: float, output_path: Path, dpi: int, dataset_name: str):
    np, plt = lazy_import_plotting()
    pred_mean, _pred_std = bundle.pipeline.predict(x_data, return_std=True)
    pred_mean = np.clip(pred_mean, 0.0, 1.0)
    pred_label = (pred_mean >= decision_threshold).astype(int)
    errors = pred_label != y_data.astype(int)
    row_index = np.arange(len(y_data))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
        constrained_layout=True,
    )
    fig.suptitle(
        f"{dataset_name} labels vs model output: errors={int(errors.sum())}/{len(errors)}, threshold={decision_threshold:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(row_index, y_data, color="#111827", linewidth=1.2, alpha=0.65, label="true label")
    ax.scatter(row_index, pred_mean, s=13, c="#dc2626", alpha=0.65, label="mean_prediction")
    ax.scatter(row_index[errors], pred_mean[errors], s=36, facecolors="none", edgecolors="#7c2d12", linewidths=1.4, label="wrong side of threshold")
    ax.axhline(decision_threshold, color="#111827", linestyle="--", linewidth=1.2, label=f"decision threshold={decision_threshold:.2f}")
    ax.set_ylabel("label / risk score")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)

    ax = axes[1]
    residual = pred_mean - y_data
    colors = np.where(errors, "#dc2626", "#2563eb")
    ax.scatter(row_index, residual, s=13, c=colors, alpha=0.68)
    ax.axhline(0.0, color="#111827", linestyle="-", linewidth=1.0)
    ax.axhline(decision_threshold, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.axhline(decision_threshold - 1.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.set_title("Residual: mean_prediction - true label")
    ax.set_xlabel(f"{dataset_name.lower()} sample index")
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def read_threshold_metrics(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"threshold metrics CSV not found: {csv_path}")
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("threshold metrics CSV is missing header")
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError("threshold metrics CSV has no rows")
    return rows


def save_threshold_selection(metrics_rows: list[dict[str, float]], output_path: Path, dpi: int):
    np, plt = lazy_import_plotting()
    decision_values = sorted({row["decision_threshold"] for row in metrics_rows})
    high_values = sorted({row["high_threshold"] for row in metrics_rows})

    best = select_best_thresholds(metrics_rows)

    best_by_decision = []
    for threshold in decision_values:
        candidates = [row for row in metrics_rows if row["decision_threshold"] == threshold]
        best_by_decision.append(
            min(
                candidates,
                key=lambda row: (
                    row["decision_cost"],
                    -row["recall"],
                    row["false_positive_rate"],
                ),
            )
        )

    cost_grid = np.full((len(high_values), len(decision_values)), np.nan)
    temporal_recall_grid = np.full_like(cost_grid, np.nan)
    for row in metrics_rows:
        x = decision_values.index(row["decision_threshold"])
        y = high_values.index(row["high_threshold"])
        cost_grid[y, x] = row["decision_cost"]
        temporal_recall_grid[y, x] = row["temporal_recall"]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    fig.suptitle(
        f"Validation threshold selection: best decision={best['decision_threshold']:.2f}, high={best['high_threshold']:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(decision_values, [row["precision"] for row in best_by_decision], marker="o", label="precision")
    ax.plot(decision_values, [row["recall"] for row in best_by_decision], marker="o", label="recall")
    ax.plot(decision_values, [row["f1"] for row in best_by_decision], marker="o", label="f1")
    ax.axvline(best["decision_threshold"], color="#111827", linestyle="--", label="selected decision")
    ax.set_title("Decision threshold tradeoff")
    ax.set_xlabel("decision_threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(decision_values, [row["fn"] for row in best_by_decision], marker="o", label="FN")
    ax.plot(decision_values, [row["fp"] for row in best_by_decision], marker="o", label="FP")
    ax.plot(decision_values, [row["decision_cost"] for row in best_by_decision], marker="o", label="weighted cost")
    ax.axvline(best["decision_threshold"], color="#111827", linestyle="--", label="selected decision")
    ax.set_title("FN / FP and weighted decision cost")
    ax.set_xlabel("decision_threshold")
    ax.set_ylabel("count / cost")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 0]
    image = ax.imshow(
        cost_grid,
        origin="lower",
        aspect="auto",
        extent=[min(decision_values), max(decision_values), min(high_values), max(high_values)],
        cmap="viridis_r",
    )
    ax.scatter([best["decision_threshold"]], [best["high_threshold"]], c="#dc2626", s=60, marker="x", label="selected")
    ax.set_title("Weighted decision cost scan")
    ax.set_xlabel("decision_threshold")
    ax.set_ylabel("high_threshold")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="decision_cost")

    ax = axes[1, 1]
    image = ax.imshow(
        temporal_recall_grid,
        origin="lower",
        aspect="auto",
        extent=[min(decision_values), max(decision_values), min(high_values), max(high_values)],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.scatter([best["decision_threshold"]], [best["high_threshold"]], c="#22c55e", s=60, marker="x", label="selected")
    ax.set_title("Temporal recall scan")
    ax.set_xlabel("decision_threshold")
    ax.set_ylabel("high_threshold")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="temporal_recall")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_gp_surface(bundle, x_data, y_data, feature_x: str, feature_y: str, grid_size: int, output_path: Path, dpi: int):
    np, plt = lazy_import_plotting()
    features = bundle.features
    ix = features.index(feature_x)
    iy = features.index(feature_y)

    x_min, x_max = range_with_margin(x_data[:, ix], 0.12)
    y_min, y_max = range_with_margin(x_data[:, iy], 0.12)
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)

    medians = np.median(x_data, axis=0)
    grid = np.tile(medians, (grid_size * grid_size, 1))
    grid[:, ix] = xx.ravel()
    grid[:, iy] = yy.ravel()

    mean, std = bundle.pipeline.predict(grid, return_std=True)
    mean = np.clip(mean, 0.0, 1.0).reshape(grid_size, grid_size)
    variance = np.square(std).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    fig.suptitle(
        f"GP 2D slice: {feature_x} x {feature_y}; other features fixed at training medians",
        fontsize=14,
        fontweight="bold",
    )

    normal = y_data == 0
    risk = y_data == 1
    panels = [
        (axes[0], mean, "Risk score surface", "viridis", 0.0, 1.0),
        (axes[1], variance, "Predictive variance surface", "magma", float(variance.min()), float(variance.max())),
    ]
    for ax, surface, title, cmap, vmin, vmax in panels:
        image = ax.imshow(
            surface,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(x_data[normal, ix], x_data[normal, iy], s=8, c="#2563eb", alpha=0.35, label="label 0")
        ax.scatter(x_data[risk, ix], x_data[risk, iy], s=8, c="#dc2626", alpha=0.35, label="label 1")
        ax.set_title(title)
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].legend(loc="upper right")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_prediction_timeline(bundle, predictions: list[dict], decision_threshold: float, high_threshold: float, output_path: Path, dpi: int):
    np, plt = lazy_import_plotting()
    row_index = np.arange(len(predictions))
    mean = np.array([float(item["mean_prediction"]) for item in predictions])
    variance = np.array([float(item["variance"]) for item in predictions])
    smoothed = np.array([
        float(item.get("temporal_risk", {}).get("smoothed_risk", item["mean_prediction"]))
        for item in predictions
    ])
    states = [item.get("temporal_risk", {}).get("state", "normal") for item in predictions]
    reliability = [item["reliability_status"] for item in predictions]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.1, 0.65]},
        constrained_layout=True,
    )
    fig.suptitle("Prediction replay: risk score, uncertainty, and temporal state", fontsize=14, fontweight="bold")

    state_colors = {
        "watch": "#fde68a",
        "warning": "#fdba74",
        "critical": "#fca5a5",
    }
    ax = axes[0]
    for idx, state in enumerate(states):
        color = state_colors.get(state)
        if color:
            ax.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=0.35, linewidth=0)
    ax.plot(row_index, mean, color="#dc2626", marker="o", linewidth=2.0, label="mean_prediction")
    ax.plot(row_index, smoothed, color="#111827", linestyle="--", linewidth=1.8, label="smoothed risk")
    ax.axhline(decision_threshold, color="#111827", linestyle=":", linewidth=1.4, label=f"decision threshold={decision_threshold:.2f}")
    ax.axhline(high_threshold, color="#b45309", linestyle=":", linewidth=1.4, label="high threshold")
    ax.set_ylabel("risk score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Risk score and temporal state bands")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1]
    ax.plot(row_index, variance, color="#7c2d12", marker="o", linewidth=2.0, label="variance")
    ax.axhline(bundle.reliable_variance_max, color="#16a34a", linestyle="--", linewidth=1.3, label="reliable max")
    ax.axhline(bundle.review_variance_min, color="#7c2d12", linestyle="--", linewidth=1.3, label="review min")
    ax.set_ylabel("variance")
    ax.set_title("Predictive variance gates")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2]
    reliability_colors = {
        "reliable": "#16a34a",
        "caution": "#f59e0b",
        "review_required": "#7c2d12",
    }
    for idx, (status, state) in enumerate(zip(reliability, states)):
        ax.scatter(idx, 0, s=95, color=reliability_colors.get(status, "#6b7280"))
        ax.text(idx, -0.35, state, ha="center", va="top", fontsize=8, rotation=35)
    ax.set_yticks([])
    ax.set_ylim(-0.75, 0.45)
    ax.set_title("Reliability marker color and temporal state text")
    ax.set_xlabel("prediction CSV row index")
    ax.grid(True, axis="x", alpha=0.20)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    if args.grid_size < 10:
        raise ValueError("grid-size must be at least 10")

    bundle = load_model(args.model)
    if args.feature_x not in bundle.features:
        raise ValueError(f"feature-x must be one of: {', '.join(bundle.features)}")
    if args.feature_y not in bundle.features:
        raise ValueError(f"feature-y must be one of: {', '.join(bundle.features)}")
    if args.feature_x == args.feature_y:
        raise ValueError("feature-x and feature-y must be different")

    train_rows = read_csv_rows(args.train_csv)
    x_data, y_data = build_dataset(train_rows, bundle.features, args.label)
    validation_data = None
    if args.valid_csv is not None:
        valid_rows = read_csv_rows(args.valid_csv)
        validation_data = build_dataset(valid_rows, bundle.features, args.label)

    threshold_metrics = None
    best_thresholds = None
    effective_decision_threshold = bundle.decision_threshold
    effective_high_threshold = args.high_threshold
    if args.threshold_metrics_csv is not None:
        threshold_metrics = read_threshold_metrics(args.threshold_metrics_csv)
        best_thresholds = select_best_thresholds(threshold_metrics)
        effective_decision_threshold = float(best_thresholds["decision_threshold"])
        effective_high_threshold = float(best_thresholds["high_threshold"])

    predict_rows = read_prediction_feature_rows(args.predict_csv, bundle.features)
    predictions = [predict_with_values(bundle, values) for values in predict_rows]
    predictions = apply_temporal_risk_filter(
        predictions,
        TemporalRiskConfig(window_size=args.window_size, high_threshold=effective_high_threshold),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        args.out_dir / "dataset_distribution.png",
        args.out_dir / ("validation_label_vs_prediction.png" if validation_data is not None else "label_vs_prediction.png"),
        args.out_dir / "gp_surface.png",
        args.out_dir / "prediction_timeline.png",
    ]
    save_training_effect(bundle, x_data, y_data, args.feature_x, args.feature_y, effective_decision_threshold, output_files[0], args.dpi)
    if validation_data is not None:
        valid_x, valid_y = validation_data
        save_label_vs_prediction(bundle, valid_x, valid_y, effective_decision_threshold, output_files[1], args.dpi, "Validation")
    else:
        save_label_vs_prediction(bundle, x_data, y_data, effective_decision_threshold, output_files[1], args.dpi, "Training")
    save_gp_surface(bundle, x_data, y_data, args.feature_x, args.feature_y, args.grid_size, output_files[2], args.dpi)
    save_prediction_timeline(bundle, predictions, effective_decision_threshold, effective_high_threshold, output_files[3], args.dpi)

    if threshold_metrics is not None:
        threshold_plot = args.out_dir / "threshold_selection.png"
        save_threshold_selection(threshold_metrics, threshold_plot, args.dpi)
        output_files.append(threshold_plot)

    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
