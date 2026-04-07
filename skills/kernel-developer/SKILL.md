---
name: kernel-developer
description: >
  Ascend kernel 开发专家 Skill（主入口）。通过开发自定义 TileLang kernel 和 AscendC kernel
  完成当前算子任务：实现 model_new_tilelang.py 和 model_new_ascendc.py。
argument-hint: >
  输入：output_dir 目录路径（包含 model.py）。
  输出：完整的 kernel 实现和验证结果。
  注意：本 Skill 协调多个子 Skill 完成全流程。
---

# Ascend Kernel 开发主 Skill

你是一名 Ascend kernel 开发专家。你的默认目标是加速 `{output_dir}/model.py` 中的 PyTorch Model，通过开发自定义 TileLang kernel 和 AscendC kernel 完成当前算子任务：实现 `{output_dir}/model_new_tilelang.py` 并调用自定义 TileLang kernel，实现 `{output_dir}/model_new_ascendc.py` 并调用 AscendC kernel。

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_tilelang.py` 和 `model_new_ascendc.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 TileLang / AscendC 实现中应尽可能避免标量逐元素写法，优先使用 `T.copy`、`T.tile.*`、矩阵/向量原语等块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计
│   │   └── tile_level/   # TileLang tile-level 设计，这里有完整可执行的 TileLang kernel
│   ├── kernel/           # 你的主要实现位置，放置 AscendC kernel
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── model_new_tilelang.py # 你的 TileLang 优化实现，调用 tile_level/ 下的 TileLang kernel
│   └── model_new_ascendc.py  # 你的 AscendC 优化实现，调用 AscendC kernel
├── docs/                 # 文档与开发指南，按需查阅，禁止修改
├── scripts/              # 远程评测与辅助脚本，禁止修改
├── utils/                # 验证、性能分析等工具，禁止修改
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

## 执行流程

本流程分为5个阶段，依次通过 subagent 执行。

### 阶段零：INPUT_CASES 精简
在执行精简之前，先将 `{output_dir}/model.py` 备份为 `{output_dir}/model.py.bak`（保留全量用例原件）。然后调用 `case-simplifier` skill 执行精简流程。该阶段完成后，`{output_dir}/model.py` 中的 `INPUT_CASES` 应已被精简为不超过 10 个代表性 case，同时覆盖所有 dtype、attribute 可选值、不同维度和极端 shape。

### 阶段一：TileLang 设计与验证
调用 `tilelang-designer` skill 执行完整流程。该阶段完成后，以下产物应已就绪：
- `{output_dir}/design/block_level/` — block-level 设计文件
- `{output_dir}/design/tile_level/` — 完整可执行的 TileLang kernel
- `{output_dir}/model_new_tilelang.py` — TileLang 优化实现，已通过 `scripts/evaluate_tilelang.sh {output_dir}` 验证

### 阶段二：AscendC 转译与验证
调用 `ascendc-translator` skill 执行完整流程。该阶段完成后，以下产物应已就绪：
- `{output_dir}/kernel/` — AscendC kernel 文件
- `{output_dir}/model_new_ascendc.py` — AscendC 优化实现，已通过 `scripts/evaluate_ascendc.sh {output_dir}` 验证

### 阶段三：性能分析（可选）
在正确性验证全部通过后，再使用 `scripts/evaluate_performance.sh {output_dir}` 做性能分析。
参考文档：`docs/PerformanceGuide.md`

### 阶段四：全量用例验证
将 `{output_dir}/model.py.bak` 恢复为 `{output_dir}/model.py`（覆盖精简后的版本，恢复全量 INPUT_CASES），然后执行 `scripts/evaluate_ascendc.sh {output_dir}` 进行一次全量用例验证。
**注意**：本阶段仅用于评估实现的完备度，只执行一次评测，**禁止对 AscendC kernel、model_new_ascendc.py 或任何其他实现文件做任何修改与修复**。无论通过与否，直接记录结果并进入下一阶段。

### 阶段五：执行 trace 记录
无论前面阶段成功或失败，都调用 `trace-recorder` skill 执行 trace 记录流程。该 skill 会在 `{output_dir}/trace.md` 生成本次任务的结构化执行记录，供后续 harness 优化使用。
