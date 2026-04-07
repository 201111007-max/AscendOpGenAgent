---
name: case-simplifier
description: >
  测试用例精简专家 Skill。读取 `{output_dir}/model.py` 中的 `INPUT_CASES`，
  对其进行精简，使 case 数量尽量不超过 10 个，同时保证覆盖度。
argument-hint: >
  输入：output_dir 目录路径。
  输出：精简后的 INPUT_CASES 已更新到 model.py 文件中。
---

# 测试用例精简 Skill

你是一名测试用例精简专家。你的目标是读取 `{output_dir}/model.py` 中的 `INPUT_CASES`，对其进行精简，使 case 数量尽量不超过 10 个，同时保证覆盖度。

## 关键限制
- 只允许修改 `{output_dir}/model.py` 中的 `INPUT_CASES` 列表，不要修改文件中的其他任何内容（Model 类、辅助函数等）。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径。

## 精简原则

精简后的 INPUT_CASES 必须满足以下覆盖要求，按优先级从高到低：

1. **dtype 覆盖**：原 INPUT_CASES 中出现的每种 tensor dtype（如 float16、float32、bfloat16 等）至少保留一个 case。
2. **attribute 可选值覆盖**：对于 `type: "attr"` 的输入，覆盖其在原 INPUT_CASES 中出现的不同取值类别（例如 bool 型的 True/False、正数/负数/零等边界值）。如果原始 attr 值变化很多，不要求每个值都保留，但要保留具有代表性的边界值。
3. **shape 维度覆盖**：覆盖原 INPUT_CASES 中出现的不同 tensor 维度数（1维、2维、3维、4维等），每种维度至少保留一个 case。
4. **shape 极端值覆盖**：保留极端小（如最小 shape）和极端大（如最大 shape）的 case。
5. **广播模式覆盖**（如适用）：如果原 INPUT_CASES 中存在 broadcasting 场景（shape 不完全一致的 tensor 对），保留至少一个 broadcasting case。

## 流程

1. 读取 `{output_dir}/model.py`，分析 `INPUT_CASES` 的完整内容。
2. 统计原始 cases 的 dtype 集合、attr 值集合、shape 维度集合、shape 大小范围、是否存在 broadcasting。
3. 按上述精简原则选取不超过 10 个代表性 case，尽量让每个 case 同时覆盖多个维度的差异。
4. 用精简后的 INPUT_CASES 替换原文件中的 INPUT_CASES，保持其他代码不变。
