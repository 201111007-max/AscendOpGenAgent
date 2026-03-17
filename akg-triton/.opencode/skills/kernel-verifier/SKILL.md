---
name: kernel-verifier
description: >
  算子代码验证 Skill — 按照标准验证流程验证生成的内核代码。
  创建验证项目文件，调用 scripts/verify.py 运行验证并收集结果。
argument-hint: >
  输入：generated-code-path、task-file-path、op-name。
  输出：验证结果（成功/失败）、错误信息、性能数据。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Kernel Verifier Skill

<role>
你是一个内核代码验证专家。你的任务是按照标准验证流程，创建验证项目并运行，检查生成的算子代码是否能正确编译运行且与参考实现的输出一致。
</role>

## 验证流程

```
输入：generated_code.py + task_file.py
    ↓
[1. 创建验证项目] → 两个文件
    ↓
[2. 执行验证脚本] → scripts/verify.py --op_name ...
    ↓
[3. 收集结果]
    ↓
输出：验证结果
```

---

## Step 1: 创建验证项目

在验证目录（如 `{output-path}/verify/`）下创建两个文件：

### 文件 1: `{op_name}_torch.py`

直接复制任务文件的完整内容。此文件包含 `Model`、`get_inputs()`、`get_init_inputs()`。

### 文件 2: `{op_name}_triton_ascend_impl.py`

直接复制生成代码的完整内容。此文件包含 `ModelNew` 类。

---

## Step 2: 执行验证

使用 `bash` 工具调用本 skill 自带的验证脚本：

```bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python3 <skill-path>/scripts/verify.py \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --timeout 300
```

**实际调用示例**（假设验证目录为 `/tmp/workspace/softmax/verify`，算子名为 `softmax`）：

```bash
python3 /path/to/kernel-verifier/scripts/verify.py \
    --op_name softmax \
    --verify_dir /tmp/workspace/softmax/verify \
    --timeout 300
```

**参数说明**：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--op_name` | 是 | 算子名称，与文件名前缀对应 |
| `--verify_dir` | 否 | 验证目录路径，默认当前目录 |
| `--timeout` | 否 | 超时秒数，默认 300 |

**超时设置**：默认 300 秒，复杂算子可适当增加。

---

## Step 3: 收集结果

根据脚本的退出码和输出判断验证结果：

### 验证通过

脚本 stdout 输出 `"验证成功"` 且退出码为 0。

返回：
- `verifier_result = true`
- `verifier_error = ""`

### 验证失败

脚本 stderr 包含错误信息且退出码非 0。

返回：
- `verifier_result = false`
- `verifier_error` = stderr 中的完整错误输出（包括 AssertionError 信息和 traceback）

### 超时

脚本输出 `"验证超时"` 且退出码为 1。

返回：
- `verifier_result = false`
- `verifier_error = "验证超时（300秒）"`

---

## 精度阈值说明

验证使用基于数据类型的**相对误差**比较，与 `torch.allclose` 不同：

| 数据类型 | 精度阈值 (limit) | 说明 |
|---------|-----------------|------|
| `float16` | 0.004 | 半精度浮点 |
| `bfloat16` | 0.03 | BF16 精度较低 |
| `int8` | 0.01 | 整数量化 |
| 其他（float32 等） | 0.02 | 默认阈值 |

**比较规则**：
1. 形状必须一致
2. NaN 位置必须一致
3. Inf 位置和符号必须一致
4. 有限值：计算相对误差，超过阈值的数量不得超过 `有限值总数 × limit`

---

## 验证脚本位置

验证脚本位于本 skill 的 `scripts/verify.py`，支持以下 CLI 参数：
- `--op_name`：算子名称（必填）
- `--verify_dir`：验证目录路径（默认 `.`）
- `--timeout`：超时秒数（默认 300）
