#!/usr/bin/env python3
"""训练小车卡死/脱落高斯过程模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gp_car_stuck_model_lib import (
    DEFAULT_CALIBRATION_MODE,
    DEFAULT_FEATURES,
    DEFAULT_LABEL,
    DEFAULT_RELIABLE_QUANTILE,
    DEFAULT_REVIEW_QUANTILE,
    print_json,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练高斯过程小车卡死/脱落模型")
    parser.add_argument("--csv", required=True, type=Path, help="训练数据 CSV 路径")
    parser.add_argument("--model-out", required=True, type=Path, help="输出模型文件路径")
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="输入特征列名，默认: drive_current speed acceleration",
    )
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名，默认: label")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="均值判定阈值")
    parser.add_argument(
        "--variance-calibration",
        choices=["quantile", "manual"],
        default=DEFAULT_CALIBRATION_MODE,
        help="方差门控阈值标定方式，默认 quantile",
    )
    parser.add_argument(
        "--reliable-quantile",
        type=float,
        default=DEFAULT_RELIABLE_QUANTILE,
        help="quantile 模式下，可靠阈值对应的验证集方差分位数",
    )
    parser.add_argument(
        "--review-quantile",
        type=float,
        default=DEFAULT_REVIEW_QUANTILE,
        help="quantile 模式下，复核阈值对应的验证集方差分位数",
    )
    parser.add_argument(
        "--reliable-variance-max",
        type=float,
        default=None,
        help="manual 模式下，方差小于等于该值时认为可靠",
    )
    parser.add_argument(
        "--review-variance-min",
        type=float,
        default=None,
        help="manual 模式下，方差大于等于该值时需要人工复核",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = train_model(
        csv_path=args.csv,
        model_path=args.model_out,
        features=args.features,
        label_name=args.label,
        decision_threshold=args.decision_threshold,
        calibration_mode=args.variance_calibration,
        reliable_variance_max=args.reliable_variance_max,
        review_variance_min=args.review_variance_min,
        reliable_quantile=args.reliable_quantile,
        review_quantile=args.review_quantile,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("训练完成")
    print_json(metrics)
    print(f"模型已保存: {args.model_out}")


if __name__ == "__main__":
    main()
