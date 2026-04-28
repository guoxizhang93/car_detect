#!/usr/bin/env python3
"""高斯过程小车卡死/脱落模型公共库。"""

from __future__ import annotations

import csv
import json
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_FEATURES = ("drive_current", "speed", "acceleration")
DEFAULT_LABEL = "label"
DEFAULT_CALIBRATION_MODE = "quantile"
DEFAULT_RELIABLE_QUANTILE = 0.50
DEFAULT_REVIEW_QUANTILE = 0.95
MIN_TRAIN_SAMPLES = 2
BINARY_LABELS = {0.0, 1.0}
REQUIRED_BINARY_LABELS = {0, 1}


@dataclass
class ModelBundle:
    features: tuple[str, ...]
    decision_threshold: float
    reliable_variance_max: float
    review_variance_min: float
    calibration_mode: str
    reliable_quantile: float | None
    review_quantile: float | None
    pipeline: object


class MissingDependencyError(RuntimeError):
    pass


def lazy_import_ml():
    try:
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        from sklearn.metrics import mean_absolute_error, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "缺少依赖，请先安装: pip install numpy scikit-learn"
        ) from exc

    return {
        "np": np,
        "GaussianProcessRegressor": GaussianProcessRegressor,
        "ConstantKernel": ConstantKernel,
        "RBF": RBF,
        "WhiteKernel": WhiteKernel,
        "mean_absolute_error": mean_absolute_error,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"training data not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头")
        return list(reader)


def build_dataset(
    rows: Sequence[dict[str, str]],
    features: Sequence[str],
    label_name: str,
):
    ml = lazy_import_ml()
    np = ml["np"]

    if not features:
        raise ValueError("at least one input feature is required")

    x_values = []
    y_values = []
    for index, row in enumerate(rows, start=2):
        try:
            feature_row = [float(row[name]) for name in features]
            label = float(row[label_name])
        except KeyError as exc:
            raise KeyError(f"CSV 缺少列: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {index} 行存在无法转换为数值的内容: {row}") from exc

        if label not in BINARY_LABELS:
            raise ValueError(f"第 {index} 行 label 必须是二分类 0 或 1: {label}")

        x_values.append(feature_row)
        y_values.append(label)

    if len(x_values) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"training data requires at least {MIN_TRAIN_SAMPLES} samples")

    return np.array(x_values, dtype=float), np.array(y_values, dtype=float)


def validate_binary_class_coverage(y_data):
    labels = set(y_data.astype(int).tolist())
    if labels != REQUIRED_BINARY_LABELS:
        missing = sorted(REQUIRED_BINARY_LABELS - labels)
        raise ValueError(
            "training data must contain both label classes 0 and 1; "
            f"missing: {', '.join(str(label) for label in missing)}"
        )


def create_pipeline(feature_count: int, random_state: int):
    ml = lazy_import_ml()
    ConstantKernel = ml["ConstantKernel"]
    RBF = ml["RBF"]
    WhiteKernel = ml["WhiteKernel"]
    GaussianProcessRegressor = ml["GaussianProcessRegressor"]
    Pipeline = ml["Pipeline"]
    StandardScaler = ml["StandardScaler"]

    if feature_count <= 0:
        raise ValueError("feature_count must be greater than 0")

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[1.0] * feature_count, length_scale_bounds=(1e-2, 1e5))
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gp", gp),
        ]
    )


def validate_quantile(name: str, value: float):
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} 必须在 0 和 1 之间")


def validate_variance_thresholds(reliable_variance_max: float, review_variance_min: float):
    if reliable_variance_max < 0.0 or review_variance_min < 0.0:
        raise ValueError("方差阈值必须大于等于 0")
    if reliable_variance_max > review_variance_min:
        raise ValueError("reliable_variance_max 不能大于 review_variance_min")



def validate_training_options(
    decision_threshold: float,
    calibration_mode: str,
    test_size: float,
):
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1")
    if calibration_mode not in {"quantile", "manual"}:
        raise ValueError("variance_calibration only supports quantile or manual")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")


def validate_unit_interval(name: str, value: float):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_temporal_config(config: "TemporalRiskConfig"):
    if config.window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    validate_unit_interval("high_threshold", config.high_threshold)
    validate_unit_interval("critical_avg_threshold", config.critical_avg_threshold)
    validate_unit_interval("ema_alpha", config.ema_alpha)
    validate_unit_interval("enter_threshold", config.enter_threshold)
    validate_unit_interval("exit_threshold", config.exit_threshold)
    if config.enter_threshold <= config.exit_threshold:
        raise ValueError("enter_threshold must be greater than exit_threshold")
    if not 1 <= config.warning_count <= config.window_size:
        raise ValueError("warning_count must be between 1 and window_size")
    if not 1 <= config.critical_count <= config.window_size:
        raise ValueError("critical_count must be between 1 and window_size")
    if config.critical_count < config.warning_count:
        raise ValueError("critical_count must be greater than or equal to warning_count")
    if not 0 <= config.max_review_required <= config.window_size:
        raise ValueError("max_review_required must be between 0 and window_size")


def choose_stratify_labels(y_data, test_size: float):
    np = lazy_import_ml()["np"]
    labels = y_data.astype(int)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) <= 1 or counts.min() < 2:
        return None
    test_count = int(round(len(labels) * test_size))
    train_count = len(labels) - test_count
    if test_count < len(unique_labels) or train_count < len(unique_labels):
        return None
    return labels

def calibrate_variance_thresholds(pred_variance, calibration_mode: str, reliable_quantile: float, review_quantile: float):
    ml = lazy_import_ml()
    np = ml["np"]

    validate_quantile("reliable_quantile", reliable_quantile)
    validate_quantile("review_quantile", review_quantile)
    if reliable_quantile >= review_quantile:
        raise ValueError("reliable_quantile 必须小于 review_quantile")

    if calibration_mode != "quantile":
        raise ValueError("当前只支持 quantile 标定模式")

    reliable_variance_max = float(np.quantile(pred_variance, reliable_quantile))
    review_variance_min = float(np.quantile(pred_variance, review_quantile))
    validate_variance_thresholds(reliable_variance_max, review_variance_min)
    return reliable_variance_max, review_variance_min


def compute_reliability_status(variance_value: float, reliable_variance_max: float, review_variance_min: float):
    if variance_value <= reliable_variance_max:
        return "reliable", False
    if variance_value >= review_variance_min:
        return "review_required", True
    return "caution", False


def train_model(
    csv_path: Path,
    model_path: Path,
    features: Sequence[str],
    label_name: str,
    decision_threshold: float,
    calibration_mode: str,
    reliable_variance_max: float | None,
    review_variance_min: float | None,
    reliable_quantile: float,
    review_quantile: float,
    test_size: float,
    random_state: int,
):
    ml = lazy_import_ml()
    np = ml["np"]
    mean_absolute_error = ml["mean_absolute_error"]
    roc_auc_score = ml["roc_auc_score"]
    train_test_split = ml["train_test_split"]

    validate_training_options(decision_threshold, calibration_mode, test_size)

    rows = read_csv_rows(csv_path)
    x_data, y_data = build_dataset(rows, features, label_name)
    validate_binary_class_coverage(y_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=choose_stratify_labels(y_data, test_size),
    )

    pipeline = create_pipeline(feature_count=len(features), random_state=random_state)
    pipeline.fit(x_train, y_train)

    pred_mean, pred_std = pipeline.predict(x_test, return_std=True)
    pred_mean = np.clip(pred_mean, 0.0, 1.0)
    pred_variance = np.square(pred_std)
    pred_label = (pred_mean >= decision_threshold).astype(int)

    if calibration_mode == "manual":
        if reliable_variance_max is None or review_variance_min is None:
            raise ValueError("manual 模式下必须同时提供 reliable_variance_max 和 review_variance_min")
        validate_variance_thresholds(reliable_variance_max, review_variance_min)
    else:
        reliable_variance_max, review_variance_min = calibrate_variance_thresholds(
            pred_variance=pred_variance,
            calibration_mode=calibration_mode,
            reliable_quantile=reliable_quantile,
            review_quantile=review_quantile,
        )

    reliability_status = [
        compute_reliability_status(v, reliable_variance_max, review_variance_min)[0]
        for v in pred_variance
    ]

    metrics = {
        "samples": int(len(x_data)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "mae": float(mean_absolute_error(y_test, pred_mean)),
        "accuracy_by_mean": float((pred_label == y_test.astype(int)).mean()),
        "mean_variance": float(pred_variance.mean()),
        "score_kind": "clipped_gp_regression_mean",
        "score_is_calibrated_probability": False,
        "roc_auc_by_mean": float(roc_auc_score(y_test, pred_mean)) if len(set(y_test.astype(int).tolist())) > 1 else None,
        "variance_calibration": {
            "mode": calibration_mode,
            "reliable_quantile": reliable_quantile if calibration_mode == "quantile" else None,
            "review_quantile": review_quantile if calibration_mode == "quantile" else None,
            "reliable_variance_max": float(reliable_variance_max),
            "review_variance_min": float(review_variance_min),
        },
        "reliability_distribution": {
            "reliable_ratio": float(sum(s == "reliable" for s in reliability_status) / len(reliability_status)),
            "caution_ratio": float(sum(s == "caution" for s in reliability_status) / len(reliability_status)),
            "review_ratio": float(sum(s == "review_required" for s in reliability_status) / len(reliability_status)),
        },
    }

    bundle = ModelBundle(
        features=tuple(features),
        decision_threshold=decision_threshold,
        reliable_variance_max=float(reliable_variance_max),
        review_variance_min=float(review_variance_min),
        calibration_mode=calibration_mode,
        reliable_quantile=(reliable_quantile if calibration_mode == "quantile" else None),
        review_quantile=(review_quantile if calibration_mode == "quantile" else None),
        pipeline=pipeline,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(bundle, f)

    return metrics


def load_model(model_path: Path) -> ModelBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, ModelBundle):
        raise TypeError("模型文件格式不正确")
    return model


def parse_feature_payload(payload: dict, features: Sequence[str]) -> list[float]:
    try:
        return [float(payload[name]) for name in features]
    except KeyError as exc:
        raise KeyError(f"输入缺少特征: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"输入特征存在非数值内容: {payload}") from exc


def predict_with_values(bundle: ModelBundle, values: Sequence[float]):
    ml = lazy_import_ml()
    np = ml["np"]

    if len(values) != len(bundle.features):
        raise ValueError(
            f"输入特征数量不匹配, 需要 {len(bundle.features)} 个: {', '.join(bundle.features)}"
        )

    x_input = np.array([list(values)], dtype=float)
    pred_mean, pred_std = bundle.pipeline.predict(x_input, return_std=True)
    mean_value = float(np.clip(pred_mean[0], 0.0, 1.0))
    variance_value = float(pred_std[0] ** 2)
    reliability_status, need_manual_check = compute_reliability_status(
        variance_value,
        bundle.reliable_variance_max,
        bundle.review_variance_min,
    )

    return {
        "features": {name: float(value) for name, value in zip(bundle.features, values)},
        "mean_prediction": mean_value,
        "score_kind": "clipped_gp_regression_mean",
        "score_is_calibrated_probability": False,
        "variance": variance_value,
        "predicted_label_by_mean": int(mean_value >= bundle.decision_threshold),
        "decision_threshold": bundle.decision_threshold,
        "reliability_status": reliability_status,
        "need_manual_check": need_manual_check,
        "variance_gate": {
            "reliable_variance_max": bundle.reliable_variance_max,
            "review_variance_min": bundle.review_variance_min,
            "calibration_mode": bundle.calibration_mode,
        },
    }



@dataclass
class TemporalRiskConfig:
    window_size: int = 10
    high_threshold: float = 0.7
    warning_count: int = 4
    critical_count: int = 7
    critical_avg_threshold: float = 0.75
    max_review_required: int = 3
    ema_alpha: float = 0.3
    enter_threshold: float = 0.7
    exit_threshold: float = 0.45


class TemporalRiskFilter:
    def __init__(self, config: TemporalRiskConfig | None = None):
        self.config = config or TemporalRiskConfig()
        validate_temporal_config(self.config)
        self.window = deque(maxlen=self.config.window_size)
        self.smoothed_risk: float | None = None
        self.latched_high = False

    def update(self, prediction: dict) -> dict:
        mean_prediction = float(prediction["mean_prediction"])
        reliability_status = prediction.get("reliability_status", "unknown")
        if self.smoothed_risk is None:
            self.smoothed_risk = mean_prediction
        else:
            alpha = self.config.ema_alpha
            self.smoothed_risk = alpha * mean_prediction + (1.0 - alpha) * self.smoothed_risk

        if self.smoothed_risk >= self.config.enter_threshold:
            self.latched_high = True
        elif self.smoothed_risk <= self.config.exit_threshold:
            self.latched_high = False

        sample = {
            "mean_prediction": mean_prediction,
            "smoothed_risk": self.smoothed_risk,
            "reliability_status": reliability_status,
            "is_high": mean_prediction >= self.config.high_threshold,
        }
        self.window.append(sample)

        high_count = sum(item["is_high"] for item in self.window)
        review_count = sum(
            item["reliability_status"] == "review_required" for item in self.window
        )
        avg_risk = sum(item["mean_prediction"] for item in self.window) / len(self.window)
        state = "normal"
        action = "run"
        critical_reason = None

        sustained_critical_risk = (
            len(self.window) == self.config.window_size
            and high_count >= self.config.critical_count
            and avg_risk >= self.config.critical_avg_threshold
            and self.latched_high
        )

        if sustained_critical_risk:
            state = "critical"
            action = "stop_or_inspect"
            if review_count > self.config.max_review_required:
                critical_reason = "sustained_high_risk_with_high_uncertainty"
            else:
                critical_reason = "sustained_high_risk"
        elif high_count >= self.config.warning_count or self.latched_high:
            state = "warning"
            action = "warn"
        elif sample["is_high"]:
            state = "watch"
            action = "observe"

        return {
            "state": state,
            "action": action,
            "smoothed_risk": float(self.smoothed_risk),
            "window_size": self.config.window_size,
            "samples_in_window": len(self.window),
            "high_count": int(high_count),
            "review_required_count": int(review_count),
            "avg_risk": float(avg_risk),
            "latched_high": self.latched_high,
            "critical_reason": critical_reason,
        }


def apply_temporal_risk_filter(predictions: Sequence[dict], config: TemporalRiskConfig | None = None):
    risk_filter = TemporalRiskFilter(config)
    results = []
    for prediction in predictions:
        enriched = dict(prediction)
        enriched["temporal_risk"] = risk_filter.update(prediction)
        results.append(enriched)
    return results



class RealTimeRiskPredictor:
    def __init__(
        self,
        model_path: Path,
        temporal_config: TemporalRiskConfig | None = None,
        enable_temporal_filter: bool = True,
    ):
        self.bundle = load_model(model_path)
        self.enable_temporal_filter = enable_temporal_filter
        self.temporal_filter = (
            TemporalRiskFilter(temporal_config) if enable_temporal_filter else None
        )

    @property
    def features(self) -> tuple[str, ...]:
        return self.bundle.features

    def predict_values(self, values: Sequence[float]) -> dict:
        prediction = predict_with_values(self.bundle, values)
        if self.temporal_filter is not None:
            prediction = dict(prediction)
            prediction["temporal_risk"] = self.temporal_filter.update(prediction)
        return prediction

    def predict_payload(self, payload: dict) -> dict:
        values = parse_feature_payload(payload, self.bundle.features)
        return self.predict_values(values)


def predict_with_model(model_path: Path, values: Sequence[float]):
    bundle = load_model(model_path)
    return predict_with_values(bundle, values)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))
