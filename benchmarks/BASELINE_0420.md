# Baseline

## Triton代码生成

### KernelBench 评测子集列表

1. **所有Vector任务** (40个)：19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,47,48,49,51,52,53,88,89,90,91,92,93,94,95,96,98,99,100
    
        注：41,42,43,44,45,46为Pooling算子，由于NPU不支持torch接口，暂时不计入评测


2. **评测子集的所有cube和cv任务**(25个)：level1的 2,4,10,11,12,13,14,15,16,17,50,54,57,61,63,64,67,82,87 和 level2的 6,12,17,23,30,94

### Triton-ascend基线结果

**评测环境**
- 分支：triton_optimization @ 2850e32a311c80f6240b588985cde3258cafa682
- 更新时间：2026-04-20
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.5.1, PyTorch 2.9.0
- 评测范围：所有Vector任务 (40个)和评测子集的所有cube和cv任务(25个)

**1. 综合评测结果**
| 指标 | 结果 |
|------|------|
| **综合精度通过率** | 56/65 (86%) |
| **综合性能≥0.6x达标** | 43/65 (66%) |
| **综合性能≥0.8x达标** | 38/65 (58%) |


**2. Vector评测结果**
| 指标 | 结果 |
|------|------|
| **Vector精度通过率** | 40/40 (100%) |
| **Vector性能≥0.6x达标** | 34/40 (85%)  |
| **Vector性能≥0.8x达标** | 33/40 (82.5%)  |


**3. CUBE & CV评测结果**
| 指标 | 结果 |
|------|------|
| **CUBE & CV精度通过率** | 16/25 (64%)  |
| **CUBE & CV性能≥0.6x达标** |  9/25 (36%)   |
| **CUBE & CV性能≥0.8x达标** | 5/25 (20%)  |


**详细结果表**

| Level | Problem ID | 算子名称 | 算子类型 | 验收类型 | 评测子集 | PyTorch 性能(ms) | triton性能(ms) | triton优化后性能(ms) | 加速比 | 加速比(性能优化后) | 精度正确 | 性能 0.6x 达标| 性能0.8x 达标 |
|:---|:---:|---------|:-------:|:------:|:------:|-------------:|---------------:|--------:|--------:|--------:|:-------:|:-------:|:-------:|
| 1 | 19 | 19_ReLU.py | Relu | VECTOR | - | 9.3672 | 11.3815 | 9.3925 | 0.82x | 1x | ✅ | ✅ | ✅ |
| 1 | 20 | 20_LeakyReLU.py | Elementwise | VECTOR | - | 11.15 | 14.22 | 14.22 | 0.79x | - | ✅ | ✅ | ❌ |
| 1 | 21 | 21_Sigmoid.py | Elementwise | VECTOR | - | 10.4849 | 13.226 | 10.6572 | 0.99x | - | ✅ | ✅ | ✅ |
| 1 | 22 | 22_Tanh.py | Elementwise | VECTOR | - | 12.89 | 13.29 | - | 1.03x | - | ✅ | ✅ | ✅ |
| 1 | 23 | 23_Softmax.py | Reduce & Norm | VECTOR | - | 32.9 | 38.7 | - | 0.85x | - | ✅ | ✅ | ✅ |
| 1 | 24 | 24_LogSoftmax.py | Reduce & Norm | VECTOR | - | 25.95 | 19.63 | - | 1.32x | - | ✅ | ✅ | ✅ |
| 1 | 25 | 25_Swish.py | Elementwise | VECTOR | - | 27.16 | 13.32 | - | 2.04x | - | ✅ | ✅ | ✅ |
| 1 | 26 | 26_GELU_.py | Elementwise | VECTOR | - | 10.7 | 12.99 | - | 0.82x | - | ✅ | ✅ | ✅ |
| 1 | 27 | 27_SELU_.py | Elementwise | VECTOR | - | 10.45 | 11.85 | 14.56 | 0.88x | 0.72x | ✅ | ✅ | ✅ |
| 1 | 28 | 28_HardSigmoid.py | Elementwise | VECTOR | - | 9.7181 | 12.6016 | - | 0.77x | - | ✅ | ✅ | ❌ |
| 1 | 29 | 29_Softplus.py | Elementwise | VECTOR | - | 51.99 | 25.19 | 45.08 | 1.15x | - | ✅ | ✅ | ✅ |
| 1 | 30 | 30_Softsign.py | Elementwise | VECTOR | - | 37.73 | 12.78 | 11.05 | - | 3.42x | ✅ | ✅ | ✅ |
| 1 | 31 | 31_ELU.py | Elementwise | VECTOR | - | 16.91 | 13.88 | - | 0.82x | - | ✅ | ✅ | ✅ |
| 1 | 32 | 32_HardTanh.py | Elementwise | VECTOR | - | 25.12 | 13.17 | 11.09 | - | 2.26x | ✅ | ✅ | ✅ |
| 1 | 33 | 33_BatchNorm.py | Reduce & Norm | VECTOR | Y | 9.0229 | 2978.1464 | 9.7455 | 0x | 0.93x | ✅ | ✅ | ✅ |
| 1 | 34 | 34_InstanceNorm.py | Reduce & Norm | VECTOR | Y | 14.5506 | 8584.2789 | 15.4906 | 0x | 0.94x | ✅ | ✅ | ✅ |
| 1 | 35 | 35_GroupNorm_.py | Reduce & Norm | VECTOR | Y | 18.06 | 35.12 | 18.94 | - | 0.97x | ✅ | ✅ | ✅ |
| 1 | 36 | 36_RMSNorm_.py | Reduce & Norm | VECTOR | Y | 37 | 19034 | 1172 | - | 0.03x | ✅ | ❌ | ❌ |
| 1 | 37 | 37_FrobeniusNorm_.py | Reduce & Norm | VECTOR | - | 15.2651 | 21.7456 | 15.1339 | 0.7x | 1.01x | ✅ | ✅ | ✅ |
| 1 | 38 | 38_L1Norm_.py | Reduce & Norm | VECTOR | - | 34.95 | 17.47 | - | - | 2.00x | - | ✅ | ✅ |
| 1 | 39 | 39_L2Norm_.py | Reduce & Norm | VECTOR | - | 20.8 | 43.35 | 24.64 | - | 0.84x | ✅ | ✅ | ✅ |
| 1 | 40 | 40_LayerNorm.py | Reduce & Norm | VECTOR | - | 2.78 | 9.84 | 1.99 | - | 1.39x | ✅ | ✅ | ✅ |
| 1 | 41 | 41_Max_Pooling_1D.py | Reduce & Norm | VECTOR | Y | 0.0704 | 0.2406 | 0.2406 | 0.29x | - | ✅ | ❌ | ❌ |
| 1 | 42 | 42_Max_Pooling_2D.py | Reduce & Norm | VECTOR | Y | 35.74 | 6398.93 | - | 0.01x | - | ✅ | ❌ | ❌ |
| 1 | 43 | 43_Max_Pooling_3D.py | Reduce & Norm | VECTOR | Y | - | - | - | - | - | ✅ | ❌ | ❌ |
| 1 | 44 | 44_Average_Pooling_1D.py | Reduce & Norm | VECTOR | Y | 28.53 | 2148.08 | - | 0.01x | - | ✅ | ❌ | ❌ |
| 1 | 45 | 45_Average_Pooling_2D.py | Reduce & Norm | VECTOR | Y | - | - | - | - | - | ✅ | ❌ | ❌ |
| 1 | 46 | 46_Average_Pooling_3D.py | Reduce & Norm | VECTOR | Y | 173.14 | 13527.78 | 13527.78 | - | 0.013x | ✅ | ❌ | ❌ |
| 1 | 47 | 47_Sum_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | - | 7.08 | 413.57 | 6.69 | - | 1.06x | ✅ | ✅ | ✅ |
| 1 | 48 | 48_Mean_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 32.84 | 23.79 | 23.79 | - | 1.38x | ✅ | ✅ | ✅ |
| 1 | 49 | 49_Max_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | - | 18.11 | 39.713 | 36.92 | 0.46x | 0.49x | ✅ | ❌ | ❌ |
| 1 | 51 | 51_Argmax_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 35.92 | 101.07 | - | 0.36x | - | ✅ | ✅ | ❌ |
| 1 | 52 | 52_Argmin_over_a_dimension.py | Reduce & Norm | VECTOR | - | 63.61 | 473.31 | 42.39 | 0.13x | 1.5x | ✅ | ✅ | ✅ |
| 1 | 53 | 53_Min_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 18.09 | 16.63 | 16.63 | 1.09x | 1.09x | ✅ | ✅ | ✅ |
| 1 | 88 | 88_MinGPTNewGelu.py | Elementwise | VECTOR | - | 4.19 | 0.61 | 0.47 | 6.85x | 13.03x | ✅ | ✅ | ✅ |
| 1 | 89 | 89_cumsum.py | Scan & Loss | VECTOR | - | 70.32 | 333.99 | 325.93 | 0.21x | 0.22x | ✅ | ❌ | ❌ |
| 1 | 90 | 90_cumprod.py | Scan & Loss | VECTOR | - | 17986.45 | 2980.72 | 2985.05 | 6.03x | 6.03x | ✅ | ✅ | ✅ |
| 1 | 91 | 91_cumsum_reverse.py | Scan & Loss | VECTOR | - | 1330.8 | 341.47 | 681.77 | 3.90x | 1.95x | ✅ | ✅ | ✅ |
| 1 | 92 | 92_cumsum_exclusive.py | Scan & Loss | VECTOR | - | 117.45 | 344.52 | 491.06 | 0.34x | 0.24x | ✅ | ❌ | ❌ |
| 1 | 93 | 93_masked_cumsum.py | Scan & Loss | VECTOR | - | 85.38 | 2976.83 | - | 0.03x | - | ✅ | ❌ | ❌ |
| 1 | 94 | 94_MSELoss | Scan & Loss | VECTOR | - | 20.96 | 6.96 | 6.96 | 3.01x | 3.01x | ✅ | ✅ | ✅ |
| 1 | 95 | 95_CrossEntropyLoss | Scan & Loss | VECTOR | - | 1.23 | 0.71 | 0.72 | 1.75x | 1.72x | ✅ | ✅ | ✅ |
| 1 | 96 | 96_HuberLoss | Scan & Loss | VECTOR | - | 14.13 | 18.62 | 7.67 | 0.76x | 1.84x | ✅ | ✅ | ✅ |
| 1 | 98 | 98_KLDivLoss.py | Scan & Loss | VECTOR | - | 7.04 | 3.24 | 2.51 | 2.17x | 2.80x | ✅ | ✅ | ✅ |
| 1 | 99 | 99_TripletMarginLoss.py | Scan & Loss | VECTOR | Y | 10.68 | 4.52 | 2.28 | 2.36x | 4.68x | ✅ | ✅ | ✅ |
| 1 | 100 | 100_HingeLoss.py | Scan & Loss | VECTOR | Y | 33.92 | 7.77 | 2072.77 | 4.37x | 0.02x | ✅ | ✅ | ✅ |
| 1 | 2 | 2_Standard_matrix_multiplication_.py | MatMul | CUBE | Y | 1.4503 | 1.6552 | 1.5584 | 0.88x | 0.93x | ✅ | ✅ | ✅ |
| 1 | 4 | 4_Matrix_vector_multiplication_.py | MatMul | CUBE | Y | 33.195 | 153.6719 | 210.2235 | 0.22x | 0.16x | ✅ | ❌ | ❌ |
| 1 | 10 | 10_3D_tensor_matrix_multiplication.py | MatMul | CUBE | Y | 0.5293 | 0.7751 | 0.7197 | 0.68x | 0.74x | ✅ | ✅ | ❌ |
| 1 | 11 | 11_4D_tensor_matrix_multiplication.py | MatMul | CUBE | Y | 5.0292 | 884.7962 | 5.4096 | 0.01x | 0.93x | ✅ | ✅ | ✅ |
| 1 | 12 | 12_Matmul_with_diagonal_matrices_.py | MatMul | CUBE | Y | 0.036 | 0.0652 | 0.0309 | 0.55x | 1.17x | ✅ | ✅ | ✅ |
| 1 | 13 | 13_Matmul_for_symmetric_matrices.py | MatMul | CUBE | Y | 1.4519 | 1.9097 | 1.5782 | 0.76x | 0.92x | ✅ | ✅ | ✅ |
| 1 | 14 | 14_Matmul_for_upper_triangular_matrices.py | MatMul | CUBE | Y | 1.5232 | 23.6837 | 6.2024 | 0.06x | 0.25x | ✅ | ❌ | ❌ |
| 1 | 15 | 15_Matmul_for_lower_triangular_matrices.py | MatMul | CUBE | Y | 1.5183 | 51.3792 | 103.5366 | 0.03x | 0.01x |✅|❌|❌|
| 1 | 16 | 16_Matmul_with_transposed_A.py | MatMul | CUBE | Y | 1.4501 | 2.084 | 1.5076 | 0.70x | 0.96x | ✅ | ✅ | ✅ |
| 1 | 17 | 17_Matmul_with_transposed_B.py | MatMul | CUBE | Y | 1.453 | 4.3877 | 1.8599 | 0.33x | 0.78x | ✅ | ✅ | ❌ |
| 1 | 50 | 50_conv_standard_2D__square_input__square_kernel | Reduce & Norm | CUBE | Y | 2.5868 | 40.6679 | 38.587 | 0.06x | 0.07x | ❌ | ❌ | ❌ |
| 1 | 54 | 54_conv_standard_3D__square_input__square_kernel.py | Conv | CUBE | Y | 5.0729 | 15484.87 | 209.08 | 0.00x | 0.02x | ✅ | ❌ | ❌ |
| 1 | 57 | 57_conv_transposed_2D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 61 | 61_conv_transposed_3D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 63 | 63_conv_standard_2D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 64 | 64_conv_transposed_1D.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 67 | 67_conv_standard_1D.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 82 | 82_conv_depthwise_2D_square_input_square_kernel.py | Conv | CUBE | Y | 5.6333 | 763.7151 | 26.0708 | 0.01x | 0.22x | ✅ | ❌ | ❌ |
| 1 | 87 | 87_conv_pointwise_2D.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 2 | 6 | 6_Conv3d_Softmax_MaxPool_MaxPool.py | level2融合 | CV | Y | 1.4386 | 8105.7984 | 8163.3078 | 0.00x | 0.00x | ✅ | ❌ | ❌ |
| 2 | 12 | 12_Gemm_Multiply_LeakyReLU.py | level2融合 | CV | Y | 1.6423 | 26.5232 | 2.2062 | 0.06x | 0.74x | ✅ | ✅ | ❌ |
| 2 | 17 | 17_Conv2d_InstanceNorm_Divide.py | level2融合 | CV | Y | / | / | / | / | / | ✅ | ❌ | ❌ |
| 2 | 23 | 23_Conv3d_GroupNorm_Mean.py | level2融合 | CV | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 2 | 30 | 30_Gemm_GroupNorm_Hardtanh.py | level2融合 | CV | Y | 2.2889 | 3.8172 | 3.4478 | 0.60x | 0.66x | ✅ | ✅ | ❌ |
| 2 | 94 | 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm.py | level2融合 | CV | Y | / | / | / | / | / | ❌ | ❌ | ❌ |

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
