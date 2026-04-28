# 小车卡死/脱落高斯过程预测说明

## 1. 目标

这套脚本用于根据小车运行时的状态量，预测当前样本是否存在卡死或脱落风险。

典型输入包括：

- `drive_current`：驱动器当前电流
- `speed`：小车当前速度
- `acceleration`：小车当前加速度

模型输出分为两部分：

- `mean_prediction`：主风险输出，范围 `0~1`
- `variance`：模型不确定性，用于判断当前输出是否可靠

其中：

- `0` 表示更接近正常状态
- `1` 表示更接近卡死/脱落等异常状态
- 数值越大，表示风险越高

## 2. 为什么使用高斯过程

高斯过程回归适合这个问题，主要因为它不仅能输出预测值，还能输出预测不确定性。

对于本项目，高斯过程有两个直接价值：

- 可以给出连续风险值，而不是只有硬分类结果
- 可以给出当前样本的预测方差，用于衡量模型对这个输入是否“熟悉”

这很适合设备监控和预警场景，因为工程上通常不仅要知道“判成什么”，还要知道“这次判定靠不靠谱”。

## 3. 核心原理

### 3.1 均值负责风险判断

模型对输入样本 `x` 会输出裁剪后的 GP 回归均值：

- `mean_prediction in [0, 1]`

这个值作为主要风险分数和判断依据。它不是经过概率校准的异常概率：

- 均值越接近 `0`，越偏向正常
- 均值越接近 `1`，越偏向卡死/脱落风险

部署时通过阈值 `decision_threshold` 做主判定：

- `mean_prediction >= decision_threshold` 判为高风险
- `mean_prediction < decision_threshold` 判为低风险

默认阈值是 `0.5`，但应结合你的数据分布和漏报/误报成本重新设定。

### 3.2 方差负责可靠性门控

模型还会输出预测标准差 `std`，脚本中使用：

- `variance = std^2`

这里的方差不是风险值，不应直接和均值加权混合。

方差的工程含义更接近：

- 当前输入是否接近训练样本分布
- 当前模型对这个输入是否足够有把握

因此本项目采用“门控”而不是“折算”：

- `variance <= reliable_variance_max`：预测可靠，状态为 `reliable`
- `variance >= review_variance_min`：预测不可靠，需要复核，状态为 `review_required`
- 中间区域：状态为 `caution`

也就是说：

- 风险高低看均值
- 结果是否可信看方差

### 3.3 为什么不能直接把均值和方差相加

如果把均值和方差混成一个分数，会带来几个问题：

- 风险定义被污染，输出不再是纯粹的“异常风险分数”
- 模型不确定性会改变业务阈值的物理意义
- 后续分析时难以区分“真高风险”和“模型不熟悉”

更稳妥的方式是把两者职责拆开：

- 均值做业务判断
- 方差做可信度管理

## 4. 脚本结构

当前目录下共有三个文件：

- [gp_car_stuck_model_lib.py]：公共模型库
- [gp_train_car_stuck_model.py]：训练脚本
- [gp_predict_car_stuck_risk.py]：部署预测脚本

职责分离如下：

- 训练脚本只负责读取数据、训练模型、标定方差阈值、保存模型
- 预测脚本只负责加载已训练模型并执行推理
- 部署环境不需要重新训练

## 5. 依赖安装

运行前需要安装：

```bash
pip install numpy scikit-learn
```


## 5.1 scikit-learn GP 与 gpytorch/GPU 对比

当前实现使用 `scikit-learn` 的 `GaussianProcessRegressor`，没有使用 `gpytorch`。原因是当前脚本优先满足中小规模离线训练和轻量部署：

| 维度 | scikit-learn GaussianProcessRegressor | gpytorch |
| --- | --- | --- |
| GPU | 不支持 GPU，主要运行在 CPU | 支持 PyTorch/CUDA，可使用 GPU |
| 工程复杂度 | 依赖少，训练/保存/部署简单 | 依赖 PyTorch/gpytorch，训练循环和设备管理更复杂 |
| 适用规模 | 中小规模数据，精确 GP，样本增大后会明显变慢 | 更适合大规模、近似 GP、变分 GP、GPU 加速 |
| 当前需求 | 输出回归风险分数、标准差、方差门控即可满足 | 当前未使用自定义似然、深度核、大规模近似等能力 |

如果后续必须使用 GPU，建议迁移到 `gpytorch`，但需要同步改造：

1. 训练脚本：把 `GaussianProcessRegressor` 改为 `gpytorch.models.ExactGP` 或变分 GP。
2. 数据预处理：继续保存标准化参数，确保部署侧一致。
3. 模型保存：不能继续只依赖当前 `pickle Pipeline`，应保存 PyTorch `state_dict`、核函数参数、特征顺序和标准化参数。
4. 预测脚本：加载模型后将输入 tensor 放到 `cuda` 或 `cpu`，输出均值和方差。
5. 性能评估：比较 CPU sklearn、CPU gpytorch、GPU gpytorch 的单样本延迟和批量吞吐。

注意：100ms 实时单样本预测不一定能从 GPU 获益，因为 GPU 数据搬运和调度也有开销。GPU 更适合批量预测、大模型、大样本训练或复杂核函数。

## 6. 训练数据格式

训练数据使用 CSV 文件，至少包含以下列：

```csv
drive_current,speed,acceleration,label
2.1,1.2,0.05,0
6.8,0.08,-1.30,1
```

字段说明：

- `drive_current`：驱动电流
- `speed`：速度
- `acceleration`：加速度
- `label`：二分类标签，`0` 表示正常，`1` 表示卡死/脱落/异常

要求：

- 所有输入特征必须是数值
- 标签必须是 `0` 或 `1`
- 表头名称必须和脚本参数一致

## 7. 训练流程

### 7.1 最常用训练命令

```bash
python gp_train_car_stuck_model.py \
  --csv your_data.csv \
  --model-out car_gp_model.pkl
```

这个流程会完成：

1. 读取 CSV 数据
2. 切分训练集和验证集
3. 训练高斯过程模型
4. 在验证集上计算预测均值和方差
5. 自动标定方差门控阈值
6. 保存模型文件

### 7.2 主要训练参数

- `--csv`：训练数据 CSV 路径
- `--model-out`：输出模型路径
- `--features`：输入特征列名列表
- `--label`：标签列名
- `--decision-threshold`：均值判定阈值
- `--variance-calibration`：方差门控阈值标定方式
- `--test-size`：验证集比例
- `--random-state`：随机种子

## 8. 方差阈值如何标定

### 8.1 默认方案：基于验证集方差分位数自动标定

默认使用：

- `--variance-calibration quantile`

训练完成后，脚本会在验证集样本上统计预测方差分布，并自动得到两个门控阈值：

- `reliable_variance_max`
- `review_variance_min`

默认分位数是：

- `reliable_quantile = 0.50`
- `review_quantile = 0.95`

含义如下：

- 验证集中约一半样本的方差会小于等于 `reliable_variance_max`
- 只有约最靠后的 `5%` 高方差样本会落入 `review_required`

这种方法的优点是：

- 不需要人工先猜一个绝对方差值
- 阈值会跟随当前模型和当前数据集自动变化
- 更适合早期试验和迭代阶段

### 8.2 什么时候调整分位数

如果你希望更严格地定义“可靠”，可以提高可靠区门槛，例如：

```bash
python gp_train_car_stuck_model.py \
  --csv your_data.csv \
  --model-out car_gp_model.pkl \
  --variance-calibration quantile \
  --reliable-quantile 0.30 \
  --review-quantile 0.90
```

一般理解如下：

- `reliable_quantile` 越小，进入 `reliable` 的条件越严格
- `review_quantile` 越小，更多样本会被标为 `review_required`

建议按业务代价调整：

- 如果误判代价很高，建议让 `review_required` 更敏感
- 如果希望部署侧尽量少人工干预，可以适当放宽 `review_quantile`

### 8.3 手工阈值模式

如果你已经通过历史经验或现场测试拿到了稳定阈值，可以使用手工模式：

```bash
python gp_train_car_stuck_model.py \
  --csv your_data.csv \
  --model-out car_gp_model.pkl \
  --variance-calibration manual \
  --reliable-variance-max 0.02 \
  --review-variance-min 0.08
```

适用于：

- 已经有成熟现场经验
- 已做过多轮数据校验
- 希望不同批次模型使用统一门控标准

## 9. Prediction input in production

Use CSV for training data and offline replay. Do not write a CSV file for every 100ms production sample. In real-time production, keep the model loaded in memory and pass collected values to the predictor API directly.

### 9.1 Training CSV vs prediction input

Training CSV needs `label` because supervised training requires ground truth:

```csv
drive_current,speed,acceleration,label
2.2,1.25,0.06,0
6.9,0.06,-1.70,1
```

Offline prediction/replay CSV does not need `label`:

```csv
drive_current,speed,acceleration
6.9,0.06,-1.70
```

### 9.2 Real-time interface usage

In production, create one predictor when the service starts, then call it every 100ms with the latest collected values:

```python
from pathlib import Path
from gp_car_stuck_model_lib import RealTimeRiskPredictor, TemporalRiskConfig

predictor = RealTimeRiskPredictor(
    model_path=Path("car_gp_model.pkl"),
    temporal_config=TemporalRiskConfig(window_size=10),
    enable_temporal_filter=True,
)

result = predictor.predict_payload(
    {
        "drive_current": current_value,
        "speed": speed_value,
        "acceleration": acceleration_value,
    }
)
```

If your acquisition layer already keeps values in model feature order, use `predict_values()` to avoid dict construction overhead:

```python
result = predictor.predict_values([current_value, speed_value, acceleration_value])
```

The returned result contains the instant GP output and, when enabled, `temporal_risk`:

- `mean_prediction`: instant risk score.
- `variance`: model uncertainty.
- `reliability_status`: uncertainty gate result.
- `temporal_risk.state`: `normal`, `watch`, `warning`, or `critical`.

### 9.3 CSV replay for offline verification

The CLI script still supports CSV for debugging and offline replay. The demo prediction CSV has 20 rows and can also simulate a real-time stream; each row represents one timestamp:

```bash
python gp_predict_car_stuck_risk.py --model car_gp_model.pkl --input-csv gp_car_stuck_predict_demo.csv --temporal-filter --window-size 10 --high-threshold 0.7 --warning-count 4 --critical-count 7 --critical-avg-threshold 0.75
```

For a real online service, keep one `RealTimeRiskPredictor` instance alive in the process. Do not restart the CLI script for every sample, otherwise the temporal window state is lost and latency will be unnecessarily high.

### 9.4 Show required feature order

```bash
python gp_predict_car_stuck_risk.py --model car_gp_model.pkl --show-features
```

## 10. 预测结果说明

预测输出示例：

```text
{
  "features": {
    "drive_current": 5.6,
    "speed": 0.12,
    "acceleration": -1.4
  },
  "mean_prediction": 0.82,
  "score_kind": "clipped_gp_regression_mean",
  "score_is_calibrated_probability": false,
  "variance": 0.014,
  "predicted_label_by_mean": 1,
  "decision_threshold": 0.5,
  "reliability_status": "reliable",
  "need_manual_check": false,
  "variance_gate": {
    "reliable_variance_max": 0.02,
    "review_variance_min": 0.08,
    "calibration_mode": "quantile"
  }
}
```

字段说明：

- `mean_prediction`：裁剪到 `0~1` 的 GP 回归风险主分数，越大越危险，但不是校准概率
- `score_is_calibrated_probability`：当前为 `false`，表示 `mean_prediction` 不应按严格概率解释
- `variance`：当前样本不确定性
- `predicted_label_by_mean`：按均值阈值得到的二值判断
- `reliability_status`：可靠性状态
- `need_manual_check`：是否需要人工复核

推荐解释方式：

- `mean_prediction` 高且 `reliable`：高风险，且结果可信
- `mean_prediction` 高但 `review_required`：高风险趋势存在且模型不够有把握，应复核；若时序窗口内持续高风险，系统会进入 `critical`
- `mean_prediction` 低且 `reliable`：可认为当前正常
- `mean_prediction` 低但 `review_required`：不能简单认定正常，应检查数据是否超出训练分布

## 11. 训练输出指标说明

训练完成后会输出一些基础指标，例如：

- `mae`
- `accuracy_by_mean`
- `roc_auc_by_mean`
- `variance_calibration`
- `reliability_distribution`

含义如下：

- `mae`：均值预测和真实标签之间的平均绝对误差
- `accuracy_by_mean`：按均值阈值做硬判定时的准确率
- `roc_auc_by_mean`：均值分数的区分能力
- `variance_calibration`：本次训练得到的方差门控阈值
- `reliability_distribution`：验证集中三种可靠性状态的占比

这些指标不能替代现场验证，但能作为模型迭代时的第一层筛选。

## 12. 工程注意事项

### 12.1 训练数据质量比模型形式更重要

如果训练数据本身标签不准、采样不稳定或工况覆盖不足，那么高斯过程输出的均值和方差都会失真。

应尽量覆盖：

- 正常运行工况
- 轻微异常工况
- 真实卡死工况
- 脱落或失联类异常工况
- 不同负载、速度、转弯、坡度、环境干扰条件

### 12.2 方差大不等于一定异常

高方差的主要含义是：

- 模型对这个输入不熟悉

原因可能包括：

- 当前输入超出训练分布
- 传感器噪声突然增大
- 现场出现了训练时未覆盖的新工况
- 数据预处理方式与训练时不一致

因此高方差应该理解为“需要检查”，而不是直接等同于“异常”。

### 12.3 不要只看二值标签

部署时建议同时保留：

- 连续风险值 `mean_prediction`
- 可靠性状态 `reliability_status`

因为工程系统通常更适合分级响应，而不是只看 `0/1`。

例如可以做：

- 低风险且可靠：放行
- 中高风险且可靠：报警或降级
- 高风险且 `review_required`：进入检查、停机或备用逻辑
- 低风险但 `review_required`：进入人工复核或备用逻辑

### 12.4 特征顺序必须一致

如果使用 `--values` 方式预测，输入顺序必须严格和训练时一致。

更稳妥的部署方式是：

- 优先使用 CSV 输入

这样可以避免因特征顺序错误导致预测失真。

### 12.5 训练和部署的数据预处理必须一致

当前脚本内部使用了标准化，并将其和模型一起保存在 `Pipeline` 中。

这意味着：

- 训练时如何标准化
- 部署时就会自动沿用同样方式

不要在部署前额外重复做一套不同标准化，否则结果会错。

### 12.6 模型文件要和脚本版本一起管理

建议在工程中同时记录：

- 训练数据版本
- 训练脚本版本
- 模型文件版本
- 特征列表
- 决策阈值
- 方差门控阈值

否则后期很难追踪某个现场结果到底来自哪一版模型。

## 13. 推荐部署策略

一个更稳妥的现场策略通常是：

1. 用 `mean_prediction` 做主风险判断
2. 用 `variance` 做可信度门控
3. 对 `review_required` 样本走人工复核或备用规则
4. 定期回收这些高方差样本，补充进训练集重新训练

这样模型会不断扩展已知工况，方差门控也会越来越稳定。

## 14. 当前实现边界

当前实现适合：

- 离线训练
- 单样本部署预测
- 中小规模数据集
- 以风险预警和可信度门控为主的工程场景


## 15. Demo data files

This folder keeps three CSV files for the complete demo:

- `gp_car_stuck_train_demo.csv`: supervised training data with `label`. It contains 1200 labeled samples covering normal, borderline, early-risk, critical-like, and out-of-distribution-like cases.
- `gp_car_stuck_predict_demo.csv`: offline prediction and stream replay data without `label`. It covers normal, borderline, isolated spike, sustained high-risk, critical-like, out-of-distribution-like, and recovery inputs. It is long enough to exercise `watch`, `warning`, and `critical` temporal states.

Recommended full demo:

```bash
python gp_train_car_stuck_model.py --csv gp_car_stuck_train_demo.csv --model-out gp_car_stuck_demo_model.pkl
python gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --input-csv gp_car_stuck_predict_demo.csv
python gp_predict_car_stuck_risk.py --model gp_car_stuck_demo_model.pkl --input-csv gp_car_stuck_predict_demo.csv --temporal-filter --window-size 10
```
