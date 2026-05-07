# Baseline

## Triton代码生成

### NPU KernelBench 评测子集列表

**Level 0** (20 tasks)：1-20

**Level 1** (31 tasks)：1-31

**Level 2** (30 tasks)：1-30

### Triton-ascend基线结果

**评测环境**
- 分支：br_430 @ latest
- 更新时间：2026-04-30
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.5.1, PyTorch 2.9.0
- 评测范围：所有Vector任务 (Level1 和 Level2) 61个和430商发算子 ( level0 ) 20个

**1. 综合评测结果**
| 指标 | 结果 |
|------|------|
| **综合精度通过率** | 81/81 (100%) |
| **综合性能≥0.6x达标** | 50/81 (62%) |
| **综合性能≥0.8x达标** | 40/81 (49%) |


**详细结果表**

| Level | Problem ID | 算子名称 | 算子类型 | 验收类型 | 评测子集 | PyTorch 性能(ms) | triton性能(ms) | triton优化后性能(ms) | 加速比 | 加速比(性能优化后) | 精度正确 | 性能 0.6x 达标| 性能0.8x 达标 |
|:---|:---:|---------|:-------:|:------:|:------:|-------------:|---------------:|--------:|--------:|--------:|:-------:|:-------:|:-------:|
| 0 | 1 | LogicalAnd | 数学计算 | - | 0.2027 | 0.1721 | 0.1619 | 0.7076 | 0.7509 | 52/52 | ✅ | ✅ | ❌ |
| 0 | 2 | FloorDivide | 数学计算 | - | 0.3763 | 0.4883 | 0.1812 | 0.3808 | 0.6965 | 50/50 | ✅ | ✅ | ❌ |
| 0 | 3 | LogicalNot | 数学计算 | - | 0.0956 | 0.0978 | 0.0978 | 1.6031 | - | 65/65 | ✅ | ✅ | ✅ |
| 0 | 4 | LogicalOr | 数学计算 | - | 0.3753 | 0.2972 | 0.291 | 1.9387 | 2.0815 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 5 | Mish | 数学计算 | - | 0.0223 | 0.0174 | 0.01 | 1.2297 | 1.8256 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 6 | RealDiv | 数学计算 | - | 0.01 | 0.0066 | 0.0066 | 1.2139 | - | 50/50 | ✅ | ✅ | ✅ |
| 0 | 7 | Hardsigmoid | 激活函数 | - | 0.0081 | 0.01 | 0.0063 | 0.7102 | 0.9963 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 8 | HardsigmoidBackward | 激活函数 | - | 0.0065 | 0.0043 | 0.0039 | 0.9678 | 1.1474 | 72/72 | ✅ | ✅ | ✅ |
| 0 | 9 | MishBackward | 激活函数 | - | 0.0361 | 0.0198 | 0.0185 | 1.5870 | 1.6099 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 10 | Relu | 激活函数 | - | 0.0102 | 0.0084 | - | 0.7200 | - | 50/50 | ✅ | ✅ | ❌ |
| 0 | 11 | Sigmoid | 激活函数 | - | 0.0126 | 0.0066 | 0.0069 | 0.9675 | 1.0767 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 12 | Swish | 激活函数 | - | 0.0231 | 0.0141 | 2.4364 | 0.0108 | 2.4576 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 13 | Elu | 激活函数 | - | 0.0503 | / | 0.1151 | / | 2.1004 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 14 | EluBackward | 激活函数 | - | 0.0568 | / | 0.0204 | / | 3.205 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 15 | GtTensor | 数学计算 | - | 0.0237 | 0.0887 | 0.0298 | 0.2864 | 0.6393 | 50/50 | ✅ | ✅ | ❌ |
| 0 | 16 | Hardswish | 激活函数 | - | 0.0016 | 0.0026 | 0.0023 | 0.5999 | 0.6729 | 50/50 | ✅ | ✅ | ❌ |
| 0 | 17 | LogAddExp | 数学计算 | - | 0.0432 | 0.0428 | - | 1.0464 | - | 50/50 | ✅ | ✅ | ✅ |
| 0 | 18 | NeScalar | 数学计算 | - | 0.0287 | 0.0142 | 0.012 | 2.8176 | 3.0113 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 19 | Threshold | 数学计算 | - | 0.0209 | 0.2379 | 0.0051 | 0.6383 | 2.7435 | 50/50 | ✅ | ✅ | ✅ |
| 0 | 20 | Relugard | 激活函数 | - | 0.0159 | 0.0075 | 0.0074 | 1.7950 | 1.814 | 50/50 | ✅ | ✅ | ✅ |
| 1 | 1 | GELU | vector | - | 0.0316 | / | 0.0507 | / | 0.6226x | 50/50 | ✅ | ✅ | ❌ |
| 1 | 2 | SwiGLU | vector | - | 0.0302 | / | 0.0324 | / | 0.9321x | 50/50 | ✅ | ✅ | ✅ |
| 1 | 3 | Add | vector | - | 1.4386 | / | 32.4138 | / | 0.0444x | 50/50 | ✅ | ❌ | ❌ |
| 1 | 4 | Abs | vector | - | 0.0262 | / | 0.0277 | / | 0.9445x | 50/50 | ✅ | ✅ | ✅ |
| 1 | 5 | Cumsum | vector | - | 0.5846 | 1.6357 | 0.4715 | 0.3574 | 1.24x | 50/50 | ✅ | ✅ | ✅ |
| 1 | 6 | Histc | vector | - | 0.0416 | 0.0679 | / | 0.6129x | / | 15/15 | ✅ | ✅ | ❌ |
| 1 | 7 | Sum | vector | - | 0.0272 | / | 0.065 | / | 0.4181x | 44/44 | ✅ | ❌ | ❌ |
| 1 | 8 | Sort | vector | - | 0.4624 | / | 0.4638 | / | 0.9969x | 31/31 | ✅ | ✅ | ✅ |
| 1 | 9 | TopK | vector | - | 0.2989 | / | 6.9931 | / | 0.0427x | 29/29 | ✅ | ❌ | ❌ |
| 1 | 10 | LayerNorm | vector | - | 0.0805 | / | 0.1112 | / | 0.7242x | 60/60 | ✅ | ✅ | ❌ |
| 1 | 11 | GroupNorm | vector | - | 0.4337 | 10.6984 | / | 0.0405x | - | 73/73 | ✅ | ❌ | ❌ |
| 1 | 12 | Permute | vector | - | 0.0145 | 0.771 | 0.1891 | 0.0188x | 0.7713x | 149/149 | ✅ | ✅ | ❌ |
| 1 | 13 | Cat | vector | - | 4.0628 | 3.956 | / | 1.0270x | / | 51/51 | ✅ | ✅ | ✅ |
| 1 | 14 | Split | vector | - | 0.0203 | 47.4486 | / | 0.0004x | / | 57/57 | ✅ | ❌ | ❌ |
| 1 | 15 | Pad | vector | - | 0.2866 | 40.3909 | 5.0251 | 0.0071 | 0.0594 | 51/51 | ✅ | ❌ | ❌ |
| 1 | 16 | Repeat | vector | - | 0.0624 | / | 14.4755 | / | 0.0043x | 49/49 | ✅ | ❌ | ❌ |
| 1 | 17 | AdamW | vector | - | 5.1254 | 0.6135 | / | 8.3541x | / | 18/18 | ✅ | ✅ | ✅ |
| 1 | 18 | Index | vector | - | 0.0189 | / | 2.3869 | / | 0.0079x | 41/41 | ✅ | ❌ | ❌ |
| 1 | 19 | IndexPut | vector | - | 0.0651 | / | 0.2213 | / | 0.2944x | 46/46 | ✅ | ❌ | ❌ |
| 1 | 20 | Gather | vector | - | 0.2548 | / | 9.7378 | / | 0.0262x | 47/47 | ✅ | ❌ | ❌ |
| 1 | 21 | Scatter | vector | - | 32.313 | 17.6788 | 17.6788 | 1.8278 | 1.8278 | 47/47 | ✅ | ✅ | ✅ |
| 1 | 22 | Nonzero | vector | - | 62.6017 | 50.7089 | / | 1.2345x | / | 50/50 | ✅ | ✅ | ✅ |
| 1 | 23 | RepeatInterleave | vector | - | 0.0728 | / | 23.4155 | / | 0.0031x | 75/75 | ✅ | ❌ | ❌ |
| 1 | 24 | EmbeddingDenseBackward | vector | - | 994.1949 | / | 4823.5206 | / | 0.2061x | 30/30 | ✅ | ❌ | ❌ |
| 1 | 25 | NLLLoss | vector | - | 0.5876 | 0.4602 | 0.4622 | 1.2767 | 1.2675 | 50/50 | ✅ | ✅ | ✅ |
| 1 | 26 | AvgPool3d | vector | - | 0.1523 | 1.7693 | / | 0.0861x | / | 72/72 | ✅ | ❌ | ❌ |
| 1 | 27 | MaxPool3d | vector | - | 0.9754 | / | 64.5584 | - | 0.0151x | 50/50 | ✅ | ❌ | ❌ |
| 1 | 28 | Interpolate | vector | - | 22.9412 | / | 12.5022 | / | 0.0571 | 73/73 | ✅ | ❌ | ❌ |
| 1 | 29 | DynamicQuant | vector | - | 1.1846 | / | 0.7965 | - | 1.4873x | 42/42 | ✅ | ✅ | ✅ |
| 1 | 30 | NMS | vector | - | / | / | / | / | / | 50/50 | ✅ | ❌ | ❌ |
| 1 | 31 | IOU | vector | - | 0.1842 | 2.4672 | / | 0.0747x | / | 30/30 | ✅ | ❌ | ❌ |
| 2 | 1 | RotaryMul | vector | - | 0.0376 | 3.275 | / | 0.0115x | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 2 | GroupNormSwish | vector | - | 1.283 | 3.2323 | / | 0.3969x | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 3 | AdvanceStepFlashattn | vector | - | 0.2927 | 0.0528 | / | 5.5406 | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 4 | MoeComputeExpertTokens | vector | - | 4.7855 | 1.7265 | / | 2.7717 | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 5 | MoeInitRouting | vector | - | 0.0257 | / | 2.7281 | / | 0.0094 | 53/53 | ✅ | ❌ | ❌ |
| 2 | 6 | MoeFinalizeRouting | vector | - | 0.5439 | 0.1096 | / | 4.9642 | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 7 | MoeGatingTopKSoftmax | vector | - | 1.2745 | 25.7509 | / | 0.0495 | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 8 | QuantScatter | vector | - | 0.5237 | 0.5306 | / | 0.9871 | / | 57/57 | ✅ | ✅ | ✅ |
| 2 | 9 | TopKTopP | vector | - | / | / | / | / | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 10 | SwigluQuant | vector | - | / | / | / | / | / | 52/52 | ✅ | ❌ | ❌ |
| 2 | 11 | DequantSwigluQuant | vector | - | 0.1861 | 0.1242 | / | 1.0082 | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 12 | KvRmsnormRopeCache | vector | - | / | / | / | / | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 13 | InterleaveRope | vector | - | 0.0689 | 0.071 | / | 0.9697 | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 14 | AdaptiveInstanceNormalization2DBackward | vector | - | 0.14 | 0.6061 | / | 0.2311x | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 15 | AttentionSoftmaxWithSoftcappingAndDropout | vector | - | 0.0857 | 1.4747 | / | 0.0581 | / | 50/50 | ✅ | ❌ | ❌ |
| 2 | 16 | Batched2DRopePositionEncodingBackward | vector | - | 0.0609 | 0.0374 | / | 1.6301 | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 17 | EmbeddingWithInitialLayernormBackward | vector | - | 1.908 | 8.7916 | / | 0.217 | - | 50/50 | ✅ | ❌ | ❌ |
| 2 | 18 | FusedAddRmsnorm | vector | - | 0.0377 | / | 0.1227 | / | 0.3075 | 50/50 | ✅ | ❌ | ❌ |
| 2 | 19 | FusedResidualRmsNormBackward | vector | - | 0.2471 | / | 0.0915 | / | 2.7011x | 50/50 | ✅ | ✅ | ✅ |
| 2 | 20 | FusedRopeWithQkNormAndKvCacheUpdate | vector | - | 1.8829 | / | 0.1245 | / | 15.1269x | 58/58 | ✅ | ✅ | ✅ |
| 2 | 21 | GaussianTopkSparseActivation | vector | - | 0.153 | / | 0.0264 | / | 5.7903x | 50/50 | ✅ | ✅ | ✅ |
| 2 | 22 | HybridAttentionMaskPreparation | vector | - | 0.5636 | / | 10.661 | / | 0.0529x | 50/50 | ✅ | ❌ | ❌ |
| 2 | 23 | HyenaFftSizePaddingRfft | vector | - | 0.6398 | / | 6.6976 | / | 0.0955x | 49/49 | ✅ | ❌ | ❌ |
| 2 | 24 | KvCacheUpdateWithRopeBackward | vector | - | 0.5936 | / | 5.3063 | / | 0.1119x | 50/50 | ✅ | ❌ | ❌ |
| 2 | 25 | MaskedSoftmaxWithAttentionDropoutBackward | vector | - | 0.0579 | / | 0.0449 | - | 1.2905x | 51/51 | ✅ | ✅ | ✅ |
| 2 | 26 | MoeGroupScoreAggregationAndMasking | vector | - | 0.152 | 0.0232 | / | 6.5626x | - | 50/50 | ✅ | ✅ | ✅ |
| 2 | 27 | MultiMaskAttentionAggregation | vector | - | 0.1601 | / | 0.114 | / | 2.4422 | 50/50 | ✅ | ✅ | ✅ |
| 2 | 28 | MultimodalRopePositionComputationWithGridBasedIndexing | vector | - | 0.6601 | 0.4289 | / | 1.5389x | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 29 | TanhGatedResidualAddBackward | vector | - | 0.1298 | 0.0473 | / | 2.7418x | / | 50/50 | ✅ | ✅ | ✅ |
| 2 | 30 | TimeDecayExponentialStabilization | vector | - | 2.744 | 0.0694 | / | 39.5174x | / | 50/50 | ✅ | ✅ | ✅ |


## AscendC代码生成

### NPUKernelBench 评测子集列表

**Level 1** (31 tasks)：1-31

**Level 2** (30 tasks)：1-30

### AscendC基线结果

**评测环境**
- 更新时间：2026-04-15
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.0, PyTorch 2.1
- 评测范围：Level 1 (1-31) + Level 2 (1-30)
- Agent：Asc算子生成Agent @kimi2.6

**综合评测结果**
| 指标 | 结果 |
|------|------|
| **综合精度通过率** | 45/61 (73.7%) |
| **综合性能≥0.6x达标** | 21/61 (34%) |
| **综合性能≥0.8x达标** | 19/61 (31%) |



**详细结果表**

| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | PyTorch 参考延迟(ms) | 生成AscendC代码延迟(ms) | 加速比 | 最终状态 | 精度正确 | 性能 0.6x 达标 | 性能 0.8x 达标 | 备注 |
|:---|:---:|---------|:-------:|:------:|:------:|-------------:|---------------:|--------:|:-------:|:-------:|:-------:|:-------:|:---|
| 1 | 1 | GELU | VECTOR | ✅ | ✅ | 0.125 | 0.306 | 0.41 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 2 | SwiGLU | VECTOR | ✅ | ✅ | 0.179 | 0.309 | 0.58 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 3 | Add | VECTOR | ✅ | ✅ | 0.148 | 0.549 | 0.27 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 4 | Abs | VECTOR | ✅ | ✅ | 0.139 | 0.34 | 0.41 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 5 | Cumsum | VECTOR | ✅ | ❌ | 0.45 | 0.616 | 0.73 | 失败 | ❌ | ❌ | ❌ | 全量验证 47/51 |
| 1 | 6 | Histc | VECTOR | ✅ | ✅ | 0.189 | 0.756 | 0.25 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 7 | Sum | VECTOR | ✅ | ✅ | 0.132 | 0.473 | 0.28 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 8 | Sort | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | 精简用例通过，全量用例因硬件 ReduceMax N-limit 失败 |
| 1 | 9 | TopK | VECTOR | ✅ | ✅ | 0.447 | 2.214 | 0.20 | 成功 | ✅ | ❌ | ❌ | AscendC ReduceMax 对 reduce 轴长度 |
| 1 | 10 | LayerNorm | VECTOR | ✅ | ✅ | 0.247 | 0.514 | 0.48 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 11 | GroupNorm | VECTOR | ✅ | ✅ | 0.603 | 0.718 | 0.84 | 成功 | ✅ | ✅ | ✅ | 大 shape 表现- Reference: 3.740 ms- TileLang:0.888 ms (4.21x) - AscendC: 0.807 ms (4.63x) |
| 1 | 12 | Permute | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 13 | Cat | VECTOR | ✅ | ✅ | 3.36 | 0.792 | 4.24 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 14 | Split | VECTOR | ✅ | ✅ | 0.064 | 0.639 | 0.10 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 15 | Pad | VECTOR | ✅ | ✅ | 1 | 3.268 | 0.31 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 16 | Repeat | VECTOR | ✅ | ❌ | 0.216 | 1.027 | 0.21 | 失败 | ❌ | ❌ | ❌ | 部分通过 |
| 1 | 17 | AdamW | VECTOR | ✅ | ✅ | 1.566 | 1.144 | 1.25 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 18 | Index | VECTOR | ✅ | ✅ | 0.098 | 0.656 | 0.15 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 19 | IndexPut | VECTOR | ✅ | ❌ | 0.149 | 0.676 | 0.22 | 失败 | ❌ | ❌ | ❌ | 部分通过 |
| 1 | 20 | Gather | VECTOR | ✅ | ✅ | 0.633 | 1.623 | 0.39 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 21 | Scatter | VECTOR | ✅ | ✅ | 29.8 | 6.652 | 4.48 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 22 | Nonzero | VECTOR | ✅ | ✅ | 71.07 | 11.93 | 5.96 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 23 | RepeatInterleave | VECTOR | ✅ | ❌ | 0.584 | 1.771 | 0.33 | 失败 | ❌ | ❌ | ❌ | 部分通过 |
| 1 | 24 | EmbeddingDenseBackward | VECTOR | ✅ | ✅ | 3.324 | 3.246 | 1.02 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 25 | NLLLoss | VECTOR | ✅ | ✅ | 37.35 | 12.66 | 2.95 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 26 | AvgPool3d | VECTOR | ✅ | ✅ | 0.49 | 0.86 | 0.57 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 27 | MaxPool3d | VECTOR | ✅ | ✅ | 62.44 | 22.42 | 1.00 | 成功 | ✅ | ✅ | ✅ | |
| 1 | 28 | Interpolate | VECTOR | ✅ | ✅ | 0.56 | 5.6 | 0.10 | 成功 | ✅ | ❌ | ❌ | |
| 1 | 29 | DynamicQuant | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 30 | NMS | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 31 | IOU | VECTOR | ✅ | ✅ | 2.229 | 1.044 | 2.13 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 1 | RotaryMul | VECTOR | ✅ | ✅ | 0.048 | 0.095 | 0.51 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 2 | GroupNormSwish | VECTOR | ✅ | ✅ | 0.048 | 0.137 | 0.35 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 3 | AdvanceStepFlashattn | VECTOR | ✅ | ✅ | 0.038 | 0.059 | 0.64 | 成功 | ✅ | ✅ | ❌ | |
| 2 | 4 | MoeInitRouting | VECTOR | ✅ | ✅ | 0.045 | 47.42 | 0.00 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 5 | MoeComputeExpertTokens | VECTOR | ✅ | ✅ | 0.044 | 0.093 | 0.47 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 6 | MoeFinalizeRouting | VECTOR | ✅ | ✅ | 0.069 | 0.207 | 0.34 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 7 | MoeGatingTopKSoftmax | VECTOR | ✅ | ✅ | 1.178 | 124.553 | 0.01 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 8 | QuantScatter | VECTOR | ✅ | ✅ | 0.653 | 0.757 | 0.86 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 9 | TopKTopP | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 10 | SwigluQuant | VECTOR | ✅ | ✅ | 0.084 | 4.079 | 0.02 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 11 | DequantSwigluQuant | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 12 | KvRmsnormRopeCache | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 13 | InterleaveRope | VECTOR | ✅ | ✅ | 0.08 | 0.187 | 0.43 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 14 | AdaptiveInstanceNormalization2DBackward | VECTOR | ✅ | ✅ | 0.358 | 0.261 | 1.37 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 15 | AttentionSoftmaxWithSoftcappingAndDropout | VECTOR | ✅ | ✅ | 0.182 | 0.36 | 0.50 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 16 | Batched2DRopePositionEncodingBackward | VECTOR | ✅ | ✅ | 0.116 | 0.521 | 0.22 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 17 | EmbeddingWithInitialLayernormBackward | VECTOR | ✅ | ✅ | 2.281 | 2.802 | 0.81 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 18 | FusedAddRmsnorm | VECTOR | ✅ | ✅ | 0.145 | 0.098 | 1.48 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 19 | FusedResidualRmsNormBackward | VECTOR | ✅ | ✅ | 0.289 | 0.285 | 1.00 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 20 | FusedRopeWithQkNormAndKvCacheUpdate | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 21 | GaussianTopkSparseActivation | VECTOR | ✅ | ✅ | 0.748 | 0.127 | 5.89 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 22 | HybridAttentionMaskPreparation | VECTOR | ✅ | ✅ | 3.852 | 0.194 | 19.86 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 23 | HyenaFftSizePaddingRfft | VECTOR | ✅ | ✅ | 0.809 | 7.262 | 0.11 | 成功 | ✅ | ❌ | ❌ | |
| 2 | 24 | KvCacheUpdateWithRopeBackward | VECTOR | ✅ | ✅ | 1.449 | 1.501 | 0.97 | 成功 | ✅ | ✅ | ✅ | 耗时太长7841s |
| 2 | 25 | MaskedSoftmaxWithAttentionDropoutBackward | VECTOR | ✅ | ✅ | 0.108 | 0.131 | 0.82 | 成功 | ✅ | ✅ | ✅ | |
| 2 | 26 | MoeGroupScoreAggregationAndMasking | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 27 | MultiMaskAttentionAggregation | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 28 | MultimodalRopePositionComputationWithGridBasedIndexing | VECTOR | ✅ | ✅ | 1.948 | 2.095 | 0.93 | 成功 | ✅ | ✅ | ✅ | 耗时太长19031s |
| 2 | 29 | TanhGatedResidualAddBackward | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 2 | 30 | TimeDecayExponentialStabilization | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |


### 结果说明

**图例说明**
- ✅：通过/成功
- ❌：失败/未通过
- ⚠️：部分通过（存在一定问题但基本功能可用）
- \：该项测试未执行或无数据

**统计口径**
- **精度通过率**：编译通过且精度正确的算子数 / 总算子数
- **性能达标率**：有性能数据且达到阈值的算子数 / 有性能数据的算子数
- **加速比**：PyTorch参考延迟 / 生成代码延迟（值>1表示生成代码更快）

**备注分类**
- 性能测试跳过：算子功能正确但未进行性能对比测试
- 直接使用torch实现：算子实现直接调用了PyTorch原语
- UB溢出：Unified Buffer溢出，需要优化内存使用
- CCU错误：计算控制单元异常，可能涉及指令地址越界
- no_kernel：未能成功生成kernel代码
