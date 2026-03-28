# AscendOpGenAgent 贡献规范

---

## 一、PR 合入规范

### 1.1 PR 标题格式

```
[<scope>] <描述>
```

| scope | 适用场景 |
|-------|---------|
| `triton` | Triton Ascend 侧改动 |
| `ascendc` | AscendC 侧改动 |
| `benchmark` | Benchmark case / 评测逻辑 |
| `router` | op-router / 路由逻辑 |
| `infra` | CI、脚本、构建 |
| `docs` | 文档 |

示例：
- `[triton] 新增 layernorm 算子生成支持`
- `[ascendc] dsl-lowering tiling pass 优化`
- `[benchmark] NPUKernelBench level2 新增 10 case`
- `[router] op-router 增加 CUDA→Ascend 路由分支`

### 1.2 合入门禁

**Benchmark评测**：跑完 BASELINE.md 全部任务，精度性能不劣化

| 门禁 | 算子生成 | 性能优化 | 其他 |
|------|:---:|:---:|:---:|
| 通过率不退化（编译/精度 Pass 数 >= BASELINE） | 必须 | 必须 | - |
| Speedup 不退化（avg 及逐任务 >= BASELINE × 0.95） | 必须 | 必须 | - |
| 性能有提升（至少 1 个算子 Speedup 提升 >= 5%） | - | 必须 | - |

**双通路冒烟**：通过 op-router 分别调用 Triton / AscendC 侧 agent，端到端跑通（编译通过 + 精度正确）— **所有非文档 PR 必须**

> 退化豁免：PR 中写明原因 + follow-up plan，2 个 maintainer Approve 后可豁免。
> 文档 / 基础设施类 PR 仅需不影响代码功能，无需上述门禁。

### 1.3 Benchmark评测

使用 `benchmark-Scheduler` agent 在 OpenCode 中执行评测：

```text
评测KernelBench中Level 1的2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 57, 61, 63, 64, 67, 82, 87, 99, 100和Level 2的6, 12, 17, 23, 30, 94的任务,
agent_workspace是<你的AscendOpGenAgent路径>
```

评测完成后，将 `agent_report.md` 中的数据与 `benchmarks/BASELINE.md` 逐项对比，在 PR 模板中填写结果。

### 1.4 双通路冒烟测试

涉及算子生成 / 性能优化 / 框架改动的 PR，须验证 op-router 的 Triton 和 AscendC 两条通路均能端到端跑通：

- **Triton 通路**：通过 op-router 调用 `AKG-triton` Agent 生成一个基础算子（如 level1 problem1），验证编译通过 + 精度正确
- **AscendC 通路**：通过 op-router 调用 `lingxi-ascendc` Agent 生成一个基础算子（如 level1 problem1），验证编译通过 + 精度正确

两条通路都通过后在 PR 模板中标记结果。若因环境原因无法运行，须在 PR 中说明。

### 1.5 基线管理

维护 `benchmarks/BASELINE.md`，记录 main 分支最新评测结果。

**更新规则**：算子生成/性能优化 PR 合入 main 后，若性能提升须更新 BASELINE.md（日期、数据、通过率、平均 Speedup）。

---

## 二、新功能接入流程

1. **创建 Agent**：`agents/<name>.md`，按 2.1 格式
2. **创建 Skill**：`skills/<skill-name>/`，按 2.2 格式
3. **注册到 op-router**：在 `agents/op-router.md` 添加路由分支（PR 待合入，暂未上仓）
4. **添加 Benchmark Case**：至少 5 个 case
5. **提供基线数据**：跑一轮 benchmark，追加到 `benchmarks/BASELINE.md`
6. **提交 PR**：按第一节规范填写模板
