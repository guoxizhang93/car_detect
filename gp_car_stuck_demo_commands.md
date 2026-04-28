# Gaussian-process car stuck/fall-off prediction demo commands

Run these commands from the repository root.

## 1. Install dependencies

```bash
pip install numpy scikit-learn
```

## 2. Train with the demo CSV

```bash
python tools/car_stuck_detection/gp_train_car_stuck_model.py --csv tools/car_stuck_detection/gp_car_stuck_train_demo.csv --model-out tools/car_stuck_detection/gp_car_stuck_demo_model.pkl
```

## 3. Show required feature order

```bash
python tools/car_stuck_detection/gp_predict_car_stuck_risk.py --model tools/car_stuck_detection/gp_car_stuck_demo_model.pkl --show-features
```

## 4. Predict from offline CSV

```bash
python tools/car_stuck_detection/gp_predict_car_stuck_risk.py --model tools/car_stuck_detection/gp_car_stuck_demo_model.pkl --input-csv tools/car_stuck_detection/gp_car_stuck_predict_demo.csv
```

## 5. Replay stream CSV with temporal filtering

```bash
python tools/car_stuck_detection/gp_predict_car_stuck_risk.py --model tools/car_stuck_detection/gp_car_stuck_demo_model.pkl --input-csv tools/car_stuck_detection/gp_car_stuck_predict_demo.csv --temporal-filter --window-size 10 --high-threshold 0.7 --warning-count 4 --critical-count 7
```

## 6. Train with manual variance gates

```bash
python tools/car_stuck_detection/gp_train_car_stuck_model.py --csv tools/car_stuck_detection/gp_car_stuck_train_demo.csv --model-out tools/car_stuck_detection/gp_car_stuck_demo_model_manual.pkl --variance-calibration manual --reliable-variance-max 0.02 --review-variance-min 0.08
```

## 7. Train with stricter quantile gates

```bash
python tools/car_stuck_detection/gp_train_car_stuck_model.py --csv tools/car_stuck_detection/gp_car_stuck_train_demo.csv --model-out tools/car_stuck_detection/gp_car_stuck_demo_model_strict.pkl --variance-calibration quantile --reliable-quantile 0.30 --review-quantile 0.90
```

## 8. Real-time production usage

CSV is for training and offline replay. In production, keep one predictor in memory and pass sampled values directly:

```python
from pathlib import Path
from gp_car_stuck_model_lib import RealTimeRiskPredictor, TemporalRiskConfig

predictor = RealTimeRiskPredictor(
    model_path=Path("tools/car_stuck_detection/gp_car_stuck_demo_model.pkl"),
    temporal_config=TemporalRiskConfig(window_size=10),
)

result = predictor.predict_payload({
    "drive_current": current_value,
    "speed": speed_value,
    "acceleration": acceleration_value,
})
```

## 9. Demo data coverage

- `gp_car_stuck_train_demo.csv`: 1200 labeled samples covering normal, borderline, early-risk, critical-like, and out-of-distribution-like cases.
- `gp_car_stuck_predict_demo.csv`: normal, borderline, isolated spike, sustained high-risk, critical-like, out-of-distribution-like, and recovery samples; long enough for temporal filtering states.
