# Performance Guide

## 性能评测
- 使用 `utils/performance.py` 作为统一的本地性能入口；当 `model.py`、`model_new_tilelang.py`、`model_new_ascendc.py` 存在时，会按实现分别评测。
- 推荐的性能评测命令：
  `SSH_TARGET=ascend-box scripts/evaluate_performance.sh current_task`
- 用法：
  `scripts/evaluate_performance.sh [task] [impl] [warmup] [repeat] [seed]`
- `impl` 支持 `reference`、`tilelang`、`ascendc`、`all`。
