---
name: tilelang-designer
description: >
  TileLang kernel 设计与实现专家 Skill。为 PyTorch Model 设计并实现自定义 TileLang kernel：
  完成 block-level 设计、tile-level 设计，并生成 model_new_tilelang.py 调用自定义 TileLang kernel。
argument-hint: >
  输入：output_dir 目录路径（包含 model.py）。
  输出：block_level/ 设计、tile_level/ 设计、model_new_tilelang.py 实现。
---

# TileLang Kernel 设计 Skill

你是一名 TileLang kernel 设计与实现专家。你的目标是为 `{output_dir}/model.py` 中的 PyTorch Model 设计并实现自定义 TileLang kernel：完成 block-level 设计、tile-level 设计，并生成 `{output_dir}/model_new_tilelang.py` 调用自定义 TileLang kernel，最终通过验证。

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_tilelang.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 TileLang 实现中应尽可能避免标量逐元素写法，优先使用 `T.copy`、`T.tile.*`、矩阵/向量原语等块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `docs/AscendC_knowledge/` 目录及其下任何文件；该目录仅供 AscendC 阶段使用，与本阶段无关。
- 禁止读取 `docs/TileLang-AscendC-API-Mapping.md`；该文档是 TileLang 到 AscendC 的转译映射，仅供 AscendC 阶段使用，与本阶段无关。

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计
│   │   └── tile_level/   # TileLang tile-level 设计，这里有完整可执行的 TileLang kernel
│   ├── kernel/           # AscendC kernel（本阶段不涉及）
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   └── model_new_tilelang.py # 你的 TileLang 优化实现，调用 tile_level/ 下的 TileLang kernel
├── docs/                 # 文档与开发指南，按需查阅，禁止修改
├── scripts/              # 远程评测与辅助脚本，禁止修改
├── utils/                # 验证、性能分析等工具，禁止修改
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

## 流程
执行以下各步骤前，必须先阅读对应的参考文档，再开始实现、验证与迭代。

1. `Block 层级设计`
   生成 `{output_dir}/design/block_level/` 下的 block-level 设计，并同步生成 `{output_dir}/model_new_tilelang.py`。在这一步只确定 block 级任务划分、流水骨架、workspace 与同步关系，具体计算细节先标记为 `TODO(tile-level)`。
   参考文档：`docs/BlockLevelDesign.md`
2. `Tile 层级设计`
   在第一步基础上继续生成 `{output_dir}/design/tile_level/`。直接以 block-level 设计为骨架，在 tile-level 中补全各处 `TODO(tile-level)`，完成可执行的 TileLang 设计与实现。
   参考文档：`docs/TileLangAscendProgrammingGuide.md`
3. `TileLang 验证与迭代`
   调用 `scripts/evaluate_tilelang.sh {output_dir}` 验证 TileLang；如果结果不正确，参考 `docs/TileLangDebug.md` 持续迭代修改，直到通过验证。
   参考文档：`docs/TileLangDebug.md`
