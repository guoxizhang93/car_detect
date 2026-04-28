# Gaussian-process car stuck/fall-off commands

Run these commands from `D:\code\car_detect`.

This machine used:

```powershell
$PY = "D:\program\python312\python.exe"
```

If `python` is already on PATH, replace `& $PY` with `python`.

## 1. Python and dependency checks

```powershell
& $PY --version
& $PY -m pip --version
& $PY -c "import numpy, sklearn, matplotlib; print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('matplotlib', matplotlib.__version__)"
```

## 2. Install dependencies

```powershell
& $PY -m pip install --upgrade pip numpy scikit-learn matplotlib
```

## 3. Compile check

```powershell
& $PY -m py_compile gp_car_stuck_model_lib.py gp_train_car_stuck_model.py gp_predict_car_stuck_risk.py gp_visualize_car_stuck_model.py
```

## 4. Train with the demo CSV

```powershell
& $PY gp_train_car_stuck_model.py --csv gp_car_stuck_train_demo.csv --model-out gp_car_stuck_demo_model.pkl
```

## 5. Split a labeled validation set

```powershell
& $PY gp_split_validation_set.py --csv gp_car_stuck_train_demo.csv --train-out validation_demo/train.csv --valid-out validation_demo/valid.csv --valid-size 0.2 --random-state 42
```

## 6. Train on the split training set

```powershell
& $PY gp_train_car_stuck_model.py --csv validation_demo/train.csv --model-out validation_demo/model.pkl
```

## 7. Select thresholds on the validation set

```powershell
& $PY gp_select_threshold.py --model validation_demo/model.pkl --valid-csv validation_demo/valid.csv --metrics-out validation_demo/threshold_metrics.csv --top-k 10
```

## 8. Show required feature order

```powershell
& $PY gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --show-features
```

## 9. Predict from offline CSV

```powershell
& $PY gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --input-csv gp_car_stuck_predict_demo.csv
```

## 10. Replay stream CSV with temporal filtering

```powershell
& $PY gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --input-csv gp_car_stuck_predict_demo.csv --temporal-filter --window-size 10 --high-threshold 0.7 --warning-count 4 --critical-count 7 --critical-avg-threshold 0.75
```

## 11. Generate matplotlib PNG visualizations

```powershell
& $PY gp_visualize_car_stuck_model.py --model validation_demo/model.pkl --train-csv validation_demo/train.csv --valid-csv validation_demo/valid.csv --predict-csv gp_car_stuck_predict_demo.csv --threshold-metrics-csv validation_demo/threshold_metrics.csv --out-dir gp_visualizations --high-threshold 0.9
```

By default this plots `speed` on the x-axis and `drive_current` on the y-axis. The current model still uses all saved features for prediction; the 2D surface fixes other features at their training median for display.
When `--threshold-metrics-csv` is provided, the visualization uses the selected validation thresholds from that file instead of the threshold stored in the model pickle.

## 12. Train with manual variance gates

```powershell
& $PY gp_train_car_stuck_model.py --csv gp_car_stuck_train_demo.csv --model-out gp_car_stuck_demo_model_manual.pkl --variance-calibration manual --reliable-variance-max 0.02 --review-variance-min 0.08
```

## 13. Train with stricter quantile gates

```powershell
& $PY gp_train_car_stuck_model.py --csv gp_car_stuck_train_demo.csv --model-out gp_car_stuck_demo_model_strict.pkl --variance-calibration quantile --reliable-quantile 0.30 --review-quantile 0.90
```

## 14. Commands actually run in this verification

```powershell
& $PY --version
& $PY -m py_compile gp_car_stuck_model_lib.py gp_train_car_stuck_model.py gp_predict_car_stuck_risk.py gp_visualize_car_stuck_model.py gp_split_validation_set.py gp_select_threshold.py
& $PY -c "import numpy, sklearn, matplotlib; print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('matplotlib', matplotlib.__version__)"
& $PY gp_train_car_stuck_model.py --csv gp_car_stuck_train_demo.csv --model-out gp_car_stuck_demo_model.pkl
& $PY gp_split_validation_set.py --csv gp_car_stuck_train_demo.csv --train-out validation_demo/train.csv --valid-out validation_demo/valid.csv --valid-size 0.2 --random-state 42
& $PY gp_train_car_stuck_model.py --csv validation_demo/train.csv --model-out validation_demo/model.pkl
& $PY gp_select_threshold.py --model validation_demo/model.pkl --valid-csv validation_demo/valid.csv --metrics-out validation_demo/threshold_metrics.csv --top-k 5
& $PY gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --show-features
& $PY gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --input-csv gp_car_stuck_predict_demo.csv
& $PY gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --input-csv gp_car_stuck_predict_demo.csv --temporal-filter --window-size 10
& $PY gp_visualize_car_stuck_model.py --model validation_demo/model.pkl --train-csv validation_demo/train.csv --valid-csv validation_demo/valid.csv --predict-csv gp_car_stuck_predict_demo.csv --threshold-metrics-csv validation_demo/threshold_metrics.csv --out-dir gp_visualizations --high-threshold 0.9
```

## 15. Optional Windows permission command

Windows has no POSIX `chmod 777`. The closest recursive equivalent is:

```powershell
icacls D:\code\car_detect /grant Everyone:(OI)(CI)F /T
```

## 16. Real-time production usage

CSV is for training and offline replay. In production, keep one predictor in memory and pass sampled values directly:

```python
from pathlib import Path
from gp_car_stuck_model_lib import RealTimeRiskPredictor, TemporalRiskConfig

predictor = RealTimeRiskPredictor(
    model_path=Path("gp_car_stuck_demo_model.pkl"),
    temporal_config=TemporalRiskConfig(window_size=10),
)

result = predictor.predict_payload({
    "drive_current": current_value,
    "speed": speed_value,
    "acceleration": acceleration_value,
})
```

## 17. Demo data coverage

- `gp_car_stuck_train_demo.csv`: 1200 labeled samples covering normal, borderline, early-risk, critical-like, and out-of-distribution-like cases.
- `gp_car_stuck_predict_demo.csv`: normal, borderline, isolated spike, sustained high-risk, critical-like, out-of-distribution-like, and recovery samples; long enough for temporal filtering states.
