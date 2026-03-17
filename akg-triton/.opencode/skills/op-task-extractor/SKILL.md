---
name: op-task-extractor
description: >
  从用户 PyTorch/Python 代码中提取算子实现，构建为 KernelBench 格式的标准化
  单文件自包含任务（task_desc.py）。支持从独立的文件中提取 shape/dtype 信息。
argument-hint: >
  需要提供：1) 待优化的代码文件路径；
  2) 可选：shape/dtype 信息来源文件路径
---

# 算子任务提取 Skill

<role>
你是一个算子任务提取专家。你的任务是从用户提供的代码中提取出可优化的
算子部分，并将其构建为 KernelBench 格式的 task_desc.py 文件。
</role>

## 目标格式

最终生成的文件必须是 **单一自包含 Python 文件**，**仅包含以下 4 个部分**：

1. `import` 区：只允许 torch / torch.nn / 标准库
2. `class Model(nn.Module)`：包装待优化算子逻辑（含 `__init__` 和 `forward`）
3. `def get_inputs()`：返回 `forward()` 的输入参数列表
4. `def get_init_inputs()`：返回 `__init__()` 的初始化参数列表

详细格式规范见 `@references/kernelbench-format.md`

---

## 提取流程

### Step 1: 代码分析

- 读取用户提供的源代码文件
- 读取 `arch` 配置（`framework=torch`、`backend=ascend`、`dsl=triton_ascend` 为固定值）
- 识别用户指定的待优化部分——即待优化的算子/计算逻辑
- 从用户的描述或提供的文件中提取 shape, dtype 等信息
- 确定算子的输入/输出签名

### Step 2: 依赖追踪

- 分析目标代码段的依赖关系（AST 级别）
- 追踪所有被调用的自定义函数/类
- 确定需要内联的外部依赖
- 识别 import 依赖链，区分标准库/PyTorch 与自定义模块

### Step 3: 构建 task_desc.py

- 将目标算子逻辑包装到 `Model.forward()` 中
- 如果算子有初始化状态（如权重、参数），放入 `Model.__init__()`
- 将所有依赖的自定义函数内联到文件中（禁止 import 外部模块）
- 根据 shape/dtype 信息构建 `get_inputs()` 和 `get_init_inputs()`
- 如果用户未提供 shape/dtype，从代码上下文推断合理默认值
- 确保 `get_inputs()` 中的设备参数为 `device='cuda'`（或根据目标后端调整）

### Step 4: 验证

- 验证生成的 task_desc.py 是否符合 KernelBench 格式规范
- 验证方法（使用 `bash` 工具执行）：
  ```bash
  python3 -c "
  import importlib.util, torch, json
  spec = importlib.util.spec_from_file_location('task', '/abs/path/task_desc.py')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  # 检查必需组件
  assert hasattr(mod, 'Model'), 'Missing class Model'
  assert hasattr(mod, 'get_inputs'), 'Missing function get_inputs'
  assert hasattr(mod, 'get_init_inputs'), 'Missing function get_init_inputs'
  # 检查可运行性
  model = mod.Model(*mod.get_init_inputs())
  inputs = mod.get_inputs()
  output = model.forward(*inputs)
  # 检查输出有效性
  if isinstance(output, torch.Tensor):
      assert not output.isnan().any(), 'Output contains NaN'
      assert not output.isinf().any(), 'Output contains Inf'
  print('Validation passed')
  "
  ```
  将 `/abs/path/task_desc.py` 替换为实际的绝对路径。
- 如果验证不通过，根据错误信息修复并重试（最多 2 次）
- 如果验证通过，进入 Step 5 请求用户确认

### Step 5: 用户确认
- 使用 `question` 工具将完整的 task_desc.py 内容展示给用户，请求确认。
- 若用户确认 task_desc 符合预期，算子提取任务完成，否则结合用户反馈返回 Step 3 重新生成。

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 自包含 | 所有依赖函数必须内联到文件中，禁止 import 项目内模块 |
| 可执行 | `Model(*get_init_inputs()).forward(*get_inputs())` 必须直接运行 |
| 确定性 | 给定相同输入，输出必须一致 |
| 无 NaN/Inf | forward 输出不能包含 NaN 或 Inf |
| 禁止重写 | 原始函数可运行就直接复用，一行都不改 |
| 返回一致 | 返回类型/形状必须与原始实现一致 |
| 合理输入 | get_inputs 应提供合理大小的输入（不能过小或过大） |

---

## 示例

### 输入

用户说："`/path/to/model.py` 的 `matmul_with_bias` 函数有优化空间，shape 信息在 `/path/to/config.py`"

### 输出 task_desc.py

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias


def get_inputs():
    batch_size = 32
    in_features = 1024
    return [torch.randn(batch_size, in_features, device='cuda')]


def get_init_inputs():
    in_features = 1024
    out_features = 512
    return [in_features, out_features]
```
