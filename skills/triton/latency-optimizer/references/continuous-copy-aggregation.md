# 连续拷贝聚合优化（Copy Aggregation on Contiguous Memory）

## 概述

**核心洞察**：当算子的语义输出在内存布局上是连续分布时，将其从"分块独立拷贝"重构为"连续聚合拷贝"，可消除 per-block 的偏移计算、循环迭代和 kernel 调度开销，显著提升内存密集型算子的性能。

**典型场景**：
- `Split` / `Chunk`：输入张量沿某一维度被切分为多个片段，但所有片段在输入侧是连续的
- `Slice` / `Narrow`：提取连续子区间
- `Pad`（特定模式）：在某一侧填充后，有效数据仍是连续块
- `Unbind` / `TensorSplit`：沿 batch/row 维度分离，每块在行方向连续

**与现有优化点的关系**：
- 与 **优化点 7（Pass 消除合并）** 的区别：Pass 合并减少的是"多次遍历同一数据"，而本优化减少的是"对连续内存的多次小粒度拷贝"
- 与 **优化点 8（维度合并）** 的区别：维度合并是将多个逻辑维度展平为一个大维度，而本优化是在已展平的连续内存上取消不必要的分块边界
- 与 **优化点 3（分核优化）** 的协同：本优化通常伴随 grid 从多维降为 1D，program 粒度从"小块"升为"整行/整段"

---

## 触发条件

必须同时满足以下三个条件才命中：

1. **语义连续**：算子的多个输出块在输入张量中占据**连续的内存区域**（无跨步间隙）
2. **分块拷贝**：当前实现使用多个 `tl.load`/`tl.store` 对或多次 kernel 调用分别处理每个输出块
3. **grid 过细**：当前 grid 维度与"分块数量"耦合（如 `grid = (num_chunks, num_rows)`），导致 program 数量远大于物理核数

**反例（不命中）**：
- `Gather` / `IndexSelect`：输出块在输入侧不连续（离散访存）
- `Scatter` / `ScatterAdd`：写入位置不连续
- `Split` 后各 chunk 在输入侧有间隔（如非均匀 stride）

---

## 判断逻辑

1. 检查算子类型：是否为 `Split` / `Chunk` / `Slice` / `Unbind` / `Pad`（单侧）等拷贝型算子
2. 检查内存连续性：
   - 输入张量在被切分维度上是否连续（stride = 1 或切分 dim 为最后一维）
   - 所有输出块在输入侧的偏移是否满足 `offset[i+1] == offset[i] + size[i]`（无间隙）
3. 检查当前实现模式：
   - 是否存在 `for chunk_idx in range(num_chunks)` 或 `grid = (num_chunks, ...)` 的分块处理
   - 每个 program 是否只处理少量元素（如 chunk_size < 4096）
4. 若 1+2+3 同时满足 → 命中

---

## 优化动作

### 核心重构：从"分块拷贝"到"连续聚合拷贝"

**原始模式（问题代码）**：
```python
@triton.jit
def split_copy_kernel(
    x_ptr, out_ptr,
    src_stride_row, total_chunk_size,
    chunk_offsets_ptr, num_chunks, num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)  # 每个 program 只处理一个 chunk
    row = tl.program_id(1)

    if chunk_idx >= num_chunks or row >= num_rows:
        return

    # 每个 program 都要加载 chunk 偏移（额外访存）
    chunk_size = tl.load(chunk_offsets_ptr + chunk_idx + 1) \
                 - tl.load(chunk_offsets_ptr + chunk_idx)
    src_offset = tl.load(chunk_offsets_ptr + chunk_idx)
    dst_offset = src_offset

    cols = tl.arange(0, BLOCK_SIZE)
    for col_start in range(0, chunk_size, BLOCK_SIZE):
        mask = col_start + cols < chunk_size
        src_idx = row * src_stride_row + src_offset + col_start + cols
        dst_idx = row * total_chunk_size + dst_offset + col_start + cols
        val = tl.load(x_ptr + src_idx, mask=mask)
        tl.store(out_ptr + dst_idx, val, mask=mask)

# grid = (num_chunks, num_rows)  # 可能高达 (64, 16) = 1024 个 program
```

**问题分析**：
- 每个 program 只拷贝 chunk_size（如 2048）个元素，粒度太细
- 需要加载 `chunk_offsets_ptr` 计算偏移（额外全局内存访问）
- grid = 1024 远超 48 核，大量调度开销
- 每个 chunk 的循环迭代次数少，循环开销占比高

**优化后模式**：
```python
@triton.jit
def split_copy_kernel(
    x_ptr,
    out_ptr,
    src_stride_row,
    total_chunk_size,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)  # 1D grid：每个 program 处理一整行

    if row >= num_rows:
        return

    # 整行连续拷贝：src 和 dst 都是连续内存
    src_offset = row * src_stride_row
    dst_offset = row * total_chunk_size

    cols = tl.arange(0, BLOCK_SIZE)
    for col_start in range(0, total_chunk_size, BLOCK_SIZE):
        mask = col_start + cols < total_chunk_size
        src_idx = src_offset + col_start + cols
        dst_idx = dst_offset + col_start + cols
        val = tl.load(x_ptr + src_idx, mask=mask)
        tl.store(out_ptr + dst_idx, val, mask=mask)

# grid = (num_rows,)  # 最多 16 个 program，调度开销极低
```

**关键变化**：
| 维度 | 原始 | 优化后 |
|------|------|--------|
| grid | `(num_chunks, num_rows)` 2D | `(num_rows,)` 1D |
| 每个 program 处理量 | 1 个 chunk（~2048 元素） | 1 整行（~131072 元素） |
| 偏移计算 | 从 `chunk_offsets_ptr` 动态加载 | 直接计算 `row * stride` |
| 循环迭代次数 | `chunk_size / BLOCK_SIZE`（少，开销占比高） | `total_chunk_size / BLOCK_SIZE`（多，摊销开销） |
| 调度开销 | 高（grid 可达 1024） | 低（grid <= 48） |

### Host 侧配合

优化后 kernel 输出一个连续 buffer，Host 侧通过 `view` + `slice` 还原为 list：
```python
out_buffer = torch.empty(num_rows * total_chunk_size, dtype=x.dtype, device=x.device)
# ... launch kernel ...
out_2d = out_buffer.view(num_rows, total_chunk_size)
return [out_2d[:, offsets[i]:offsets[i + 1]].view(...) for i in range(num_chunks)]
```

---

## 适用算子清单

| 算子 | 连续条件 | grid 变化 | 预期收益 |
|------|---------|----------|---------|
| `Split` / `Chunk` | 切分 dim 连续，chunks 无间隙 | `(num_chunks, rows)` → `(rows,)` | 0.2x → 3x+ |
| `Slice` / `Narrow` | 操作维度连续 | `(chunks, rows)` → `(rows,)` | 类似 |
| `Unbind` | 分离维度为连续轴 | `(num_slices,)` → `(1,)` 或 Host 循环 | 中等 |
| `Pad`（单侧/双侧） | 有效数据连续，填充区在尾部 | `(blocks,)` → `(rows,)` | 中等 |

---

## 预期收益

- **性能**：
  - 消除 per-chunk 偏移加载（减少全局内存访问次数）
  - 减少 grid 大小，降低 kernel 调度开销
  - 增大每个 program 的处理量，提高向量单元利用率
  - 连续内存拷贝更易触发硬件预取和 burst 传输
- **典型提升**：
  - Split（小 chunk 多数量）：0.2x → 3.0x+
  - Slice（中等 chunk）：0.5x → 2.0x+
- **精度**：无影响（纯拷贝语义不变）

---

## 验证要求

1. **精度验证必须通过**：所有 shape 的数值与参考实现完全一致
2. **性能不劣化**：任何 case 不得出现 speedup < 1.0
3. **边界检查**：确保 `total_chunk_size` 计算正确，不越界
4. **非连续场景回退**：若检测到 chunks 不连续（有间隙），应回退到分块拷贝策略

---

## 常见陷阱

### 陷阱 1：误用于非连续场景

```python
# ❌ 错误：chunks 之间有间隙，连续拷贝会包含无效数据
sizes = [2048, 2048]  # 但输入 stride 导致中间有 padding
# 此时不能直接用整行拷贝

# ✅ 正确：先检查连续性
is_contiguous = all(
    offsets[i+1] == offsets[i] + sizes[i]
    for i in range(len(sizes))
)
if is_contiguous:
    # 使用连续聚合拷贝
else:
    # 回退到分块拷贝
```

### 陷阱 2：忽略输出 buffer 的连续性

```python
# ❌ 错误：输出 buffer 未分配为连续内存
out_buffer = torch.empty(num_rows, total_chunk_size, dtype=x.dtype, device=x.device)
# 但后续 slice 可能产生非连续视图，影响下游使用

# ✅ 正确：先分配 1D 连续 buffer，再 view
out_buffer = torch.empty(num_rows * total_chunk_size, dtype=x.dtype, device=x.device)
out_2d = out_buffer.view(num_rows, total_chunk_size)
```

### 陷阱 3：grid 降维后并行度不足

```python
# ❌ 错误：num_rows = 1，grid = (1,)，单核执行
grid = (num_rows,)  # 若 num_rows = 1，无法利用多核

# ✅ 正确：若 num_rows 太小，可考虑保留 2D grid 但按行分块
# 或结合优化点 12（Grid 形状与多路径特化）做动态 dispatch
if num_rows >= 4:
    grid = (num_rows,)
else:
    # 小行数场景：每个 program 处理多行
    grid = (min(num_rows * num_inner_blocks, num_cores),)
```

---

## 与 PR #220 的兼容性说明

本优化点作为**新增优化点 15** 插入到 `latency-optimizer/SKILL.md` 的优化点列表中，位于现有优化点 14（混合策略自动选择）之后。与 PR #220 的改动无冲突：
- PR #220 主要完善了 `constexpr_parameters.md`、`grid-dispatch-specialization.md` 等现有文档
- 本优化点独立成文（`references/continuous-copy-aggregation.md`），不修改现有优化点的定义和判断逻辑
- 在 `SKILL.md` 中仅新增一个优化点条目和参考资料索引行，不影响其他优化点的执行顺序和约束
