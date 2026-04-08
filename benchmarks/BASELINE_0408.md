# Baseline

## Triton代码生成

### KernelBench 评测子集列表

**所有Vector任务** (48个)：19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100

**评测子集的所有cube和cv任务**(24个)：level1的 2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 54, 57, 61, 63, 64, 67, 82, 87 和 level2的 6, 12, 17, 23, 30, 94

### Triton-ascend基线结果

**评测环境**
- 分支：br_claudecode @ latest
- 更新时间：2026-04-07
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.5.1, PyTorch 2.9.0
- 评测范围：所有Vector任务 (48个)和评测子集的所有cube和cv任务(24个)

**综合评测结果**
| 指标 | 结果 |
|------|------|
| **综合精度通过率** | 68/72 (94%) |
| **综合性能≥0.6x达标** | 42/72 (58%) |
| **综合性能≥0.8x达标** | 38/72 (52.8%) |
| 综合平均加速比 | 4.81x ( 注：存在最大加速比 222.29x ) |


**Vector评测结果**
| 指标 | 结果 |
|------|------|
| **Vector精度通过率** | 48/48 (100%) |
| **Vector性能≥0.6x达标** | 33/48 (68.75%) |
| **Vector性能≥0.8x达标** | 33/48 (68.75%) |
| Vector 平均加速比 | 6.14x ( 注：存在最大加速比 222.29x ) |


**详细结果表**

| Level | Problem ID | 算子名称 | 算子类型 | 验收类型 | 评测子集 | PyTorch 性能(ms) | triton性能(ms) | triton优化后性能(ms) | 加速比 | 加速比(性能优化后) | 精度正确 | 性能 0.6x 达标| 性能0.8x 达标 |
|:---|:---:|---------|:-------:|:------:|:------:|-------------:|---------------:|--------:|--------:|--------:|:-------:|:-------:|:-------:|
| 1 | 19 | 19_ReLU.py | Relu | VECTOR | | 9.33 | 11.96 | | 0.78x | 0.99x | ✅ | ✅ | ✅ |
| 1 | 20 | 20_LeakyReLU.py | Elementwise | VECTOR | | 9.37 | 15.66 | | 0.6x | 0.8x | ✅ | ✅ | ✅ |
| 1 | 21 | 21_Sigmoid.py | Elementwise | VECTOR | | 9.36 | 10.47 | | 0.89x | - | ✅ | ✅ | ✅ |
| 1 | 22 | 22_Tanh.py | Elementwise | VECTOR | | 13.1341 | 12.6526 | | 1.04x | - | ✅ | ✅ | ✅ |
| 1 | 23 | 23_Softmax.py | Reduce & Norm | VECTOR | | 15.3363 | 17.2224 | | 0.89x | 1.23x | ✅ | ✅ | ✅ |
| 1 | 24 | 24_LogSoftmax.py | Reduce & Norm | VECTOR | | 14.9328 | 11.1385 | | 1.34x | 1.46x | ✅ | ✅ | ✅ |
| 1 | 25 | 25_Swish.py | Elementwise | VECTOR | | 4.3112 | 17.9197 | | 1.36x | 1.9x | ✅ | ✅ | ✅ |
| 1 | 26 | 26_GELU_.py | Elementwise | VECTOR | | 9.3711 | 52.0375 | | 0.18x | 0.81x | ✅ | ✅ | ✅ |
| 1 | 27 | 27_SELU_.py | Elementwise | VECTOR | | 9.3456 | 13.2669 | 10.9506 | 0.70x | 0.85x | ✅ | ✅ | ✅ |
| 1 | 28 | 28_HardSigmoid.py | Elementwise | VECTOR | | 9.3049 | 10.1086 | 10.219 | 0.92x | 0.91x | ✅ | ✅ | ✅ |
| 1 | 29 | 29_Softplus.py | Elementwise | VECTOR | | 23.4335 | 14.6634 | 14.8372 | 1.6x | 1.58x | ✅ | ✅ | ✅ |
| 1 | 30 | 30_Softsign.py | Elementwise | VECTOR | | 34.0657 | 9.8571 | 9.5056 | 3.46x | 3.58x | ✅ | ✅ | ✅ |
| 1 | 31 | 31_ELU.py | Elementwise | VECTOR | | 9.3463 | 12.5098 | 10.7179 | 0.75x | 0.87x | ✅ | ✅ | ✅ |
| 1 | 32 | 32_HardTanh.py | Elementwise | VECTOR | | 21.7796 | 13.485 | 10.0745 | 1.62x | 2.21x | ✅ | ✅ | ✅ |
| 1 | 33 | 33_BatchNorm.py | Reduce & Norm | VECTOR | Y | 9.0215 | 72.356 | 72.356 | 0.12x | 0.12x | ✅ | ❌ | ❌ |
| 1 | 34 | 34_InstanceNorm.py | Reduce & Norm | VECTOR | Y | 15.4106 | 33.5952 | 11.0224 | 0.46x | 1.4x | ✅ | ✅ | ✅ |
| 1 | 35 | 35_GroupNorm_.py | Reduce & Norm | VECTOR | Y | 17.97 | 1963.11 | 1355.69 | 0.01x | 0.01x | ✅ | ❌ | ❌ |
| 1 | 36 | 36_RMSNorm_.py | Reduce & Norm | VECTOR | Y | 33.57 | 263.92 | / | 0.13x | / | ✅ | ❌ | ❌ |
| 1 | 37 | 37_FrobeniusNorm_.py | Reduce & Norm | VECTOR | | 15.2562 | 15.7896 | / | 0.97 | / | ✅ | ✅ | ✅ |
| 1 | 38 | 38_L1Norm_.py | Reduce & Norm | VECTOR | | 23.76 | 15.07 | 11.8 | 1.28 | 2.01 | ✅ | ✅ | ✅ |
| 1 | 39 | 39_L2Norm_.py | Reduce & Norm | VECTOR | | 15.38 | 16.83 | 11.84 | 1.3 | 1.42 | ✅ | ✅ | ✅ |
| 1 | 40 | 40_LayerNorm.py | Reduce & Norm | VECTOR | | 2.49 | 1.1 | / | 2.27 | / | ✅ | ✅ | ✅ |
| 1 | 41 | 41_Max_Pooling_1D.py | Reduce & Norm | VECTOR | Y | 213.88 | 2103.58 | 2103.52 | 0.1 | 0.1 | ✅ | ❌ | ❌ |
| 1 | 42 | 42_Max_Pooling_2D.py | Reduce & Norm | VECTOR | Y | 27.9157 | 14546.9719 | / | 0 | / | ✅ | ❌ | ❌ |
| 1 | 43 | 43_Max_Pooling_3D.py | Reduce & Norm | VECTOR | Y | | 6163.08 | 3923.65 | | 1.57 | ✅ | ✅ | ✅ |
| 1 | 44 | 44_Average_Pooling_1D.py | Reduce & Norm | VECTOR | Y | 18.32 | 1256.22 | 1257.92 | 0.01 | 0.01 | ✅ | ❌ | ❌ |
| 1 | 45 | 45_Average_Pooling_2D.py | Reduce & Norm | VECTOR | Y | 3449.12 | 448.11 | 309.36 | 11.15 | 1.45 | ✅ | ✅ | ✅ |
| 1 | 46 | 46_Average_Pooling_3D.py | Reduce & Norm | VECTOR | Y | 153.75 | 11133.95 | 6910.94 | 0.02 | 1.61 | ✅ | ✅ | ✅ |
| 1 | 47 | 47_Sum_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | | 6.34 | 514.5 | 452.96 | 0.01 | 1.14 | ✅ | ✅ | ✅ |
| 1 | 48 | 48_Mean_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 7.57 | 182.03 | / | 0.04 | / | ✅ | ❌ | ❌ |
| 1 | 49 | 49_Max_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | | 16.26 | 396.94 | 364.42 | 0.04 | 1.09 | ✅ | ✅ | ✅ |
| 1 | 50 | 50_conv_standard_2D__square_input__square_kernel | Reduce & Norm | VECTOR | Y | 2.5786 | 0.0116 | / | 222.29 | / | ✅ | ✅ | ✅ |
| 1 | 51 | 51_Argmax_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 18.08 | 9358.42 | 4442.38 | 0.00x | 0.00x | ✅ | ❌ | ❌ |
| 1 | 52 | 52_Argmin_over_a_dimension.py | Reduce & Norm | VECTOR | | 63.6204 | 5353.3452 | / | 0.01x | / | ✅ | ❌ | ❌ |
| 1 | 53 | 53_Min_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 18.0783 | 5322.2028 | / | 0.00x | / | ✅ | ❌ | ❌ |
| 1 | 88 | 88_MinGPTNewGelu.py | Elementwise | VECTOR | | 4.1719 | 0.7925 | / | 5.26x | / | ✅ | ✅ | ✅ |
| 1 | 89 | 89_cumsum.py | Scan & Loss | VECTOR | | 70.3172 | 2778.5388 | / | 0.03x | / | ✅ | ❌ | ❌ |
| 1 | 90 | 90_cumprod.py | Scan & Loss | VECTOR | | 17839.7 | 2778.75 | 2779.82 | 6.42x | 6.42x | ✅ | ✅ | ✅ |
| 1 | 91 | 91_cumsum_reverse.py | Scan & Loss | VECTOR | | 1331.1 | 2779.74 | 2780.76 | 0.48x | 0.48x | ✅ | ❌ | ❌ |
| 1 | 92 | 92_cumsum_exclusive.py | Scan & Loss | VECTOR | | 117.49 | 2778.63 | 2778.72 | 0.04x | 0.04x | ✅ | ❌ | ❌ |
| 1 | 93 | 93_masked_cumsum.py | Scan & Loss | VECTOR | | 84.2 | 331.6 | 327.46 | 0.25x | 0.26x | ✅ | ❌ | ❌ |
| 1 | 94 | 94_MSELoss | Scan & Loss | VECTOR | | 20.93 | 13.57 | 9.35 | 1.54x | 2.24x | ✅ | ✅ | ✅ |
| 1 | 95 | 95_CrossEntropyLoss | Scan & Loss | VECTOR | | 1.23 | 2.92 | 2.48 | 0.42x | 0.49x | ✅ | ❌ | ❌ |
| 1 | 96 | 96_HuberLoss | Scan & Loss | VECTOR | | 14.1 | 19.48 | 11.61 | 0.72x | 1.21x | ✅ | ✅ | ✅ |
| 1 | 97 | 97_CosineSimilarityLoss.py | Scan & Loss | VECTOR | | 31.65 | 20.89 | 19.59 | 1.51x | 1.62x | ✅ | ✅ | ✅ |
| 1 | 98 | 98_KLDivLoss.py | Scan & Loss | VECTOR | | 7.05 | 1.5 | / | 4.69x | / | ✅ | ✅ | ✅ |
| 1 | 99 | 99_TripletMarginLoss.py | Scan & Loss | VECTOR | Y | 10.8 | 4.46 | 3.09 | 2.40x | 3.49x | ✅ | ✅ | ✅ |
| 1 | 100 | 100_HingeLoss.py | Scan & Loss | VECTOR | Y | 33.96 | 3965.15 | 14.52 | 0.01x | 2.17x | ✅ | ✅ | ✅ |
| 1 | 2 | 2_Standard_matrix_multiplication_.py | MatMul | CUBE | Y | 1.4504 | 2.3599 | 2.2977 | 0.61x | 0.63x | ✅ | ✅ | ❌ |
| 1 | 4 | 4_Matrix_vector_multiplication_.py | MatMul | CUBE | Y | 18.4718 | 84.3304 | 84.3304 | 0.22x | 0.22x | ✅ | ❌ | ❌ |
| 1 | 10 | 10_3D_tensor_matrix_multiplication.py | MatMul | CUBE | Y | 0.531 | 1.7555 | 0.6737 | 0.3x | 0.79x | ✅ | ✅ | ❌ |
| 1 | 11 | 11_4D_tensor_matrix_multiplication.py | MatMul | CUBE | Y | 5.0838 | 1729.9622 | 42.6104 | 0.0x | 0.12x | ✅ | ❌ | ❌ |
| 1 | 12 | 12_Matmul_with_diagonal_matrices_.py | MatMul | CUBE | Y | 0.0359 | 0.0729 | 0.0316 | 0.49x | 1.14x | ✅ | ✅ | ✅ |
| 1 | 13 | 13_Matmul_for_symmetric_matrices.py | MatMul | CUBE | Y | 1.2923 | 1.2896 | / | 1.0x | / | ✅ | ✅ | ✅ |
| 1 | 14 | 14_Matmul_for_upper_triangular_matrices.py | MatMul | CUBE | Y | 1.5261 | 4.4984 | 1.9458 | 0.34x | 0.77x | ✅ | ✅ | ❌ |
| 1 | 15 | 15_Matmul_for_lower_triangular_matrices.py | MatMul | CUBE | Y | 1.5271 | 1.6009 | 1.5373 | 0.95x | 0.99x | ✅ | ✅ | ✅ |
| 1 | 16 | 16_Matmul_with_transposed_A.py | MatMul | CUBE | Y | 1.45 | 1.7231 | 1.7259 | 0.84x | 0.84x | ✅ | ✅ | ✅ |
| 1 | 17 | 17_Matmul_with_transposed_B.py | MatMul | CUBE | Y | 1.3727 | / | / | / | / | ✅ | ❌ | ❌ |
| 1 | 54 | 54_conv_standard_3D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ✅ | ❌ | ❌ |
| 1 | 57 | 57_conv_transposed_2D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ✅ | ❌ | ❌ |
| 1 | 61 | 61_conv_transposed_3D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ✅ | ❌ | ❌ |
| 1 | 63 | 63_conv_standard_2D__square_input__square_kernel.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 64 | 64_conv_transposed_1D.py | Conv | CUBE | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 1 | 67 | 67_conv_standard_1D.py | Conv | CUBE | Y | / | / | / | / | / | ✅ | ❌ | ❌ |
| 1 | 82 | 82_conv_depthwise_2D_square_input_square_kernel.py | Conv | CUBE | Y | 5.62 | 633.19 | / | 0.01x | / | ✅ | ❌ | ❌ |
| 1 | 87 | 87_conv_pointwise_2D.py | Conv | CUBE | Y | 23.74 | 458.2 | / | 0.05x | / | ✅ | ❌ | ❌ |
| 2 | 6 | 6_Conv3d_Softmax_MaxPool_MaxPool.py | level2融合 | CV | Y | 1.4361 | 2034.8614 | / | 0.0x | （未做优化） | ✅ | ❌ | ❌ |
| 2 | 12 | 12_Gemm_Multiply_LeakyReLU.py | level2融合 | CV | Y | 1.6224 | 3734.2337 | 3731.9326 | 0.0x | 0.0x | ✅ | ❌ | ❌ |
| 2 | 17 | 17_Conv2d_InstanceNorm_Divide.py | level2融合 | CV | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 2 | 23 | 23_Conv3d_GroupNorm_Mean.py | level2融合 | CV | Y | / | / | / | / | / | ❌ | ❌ | ❌ |
| 2 | 30 | 30_Gemm_GroupNorm_Hardtanh.py | level2融合 | CV | Y | 2.0097 | 2.3768 | 1.7603 | 0.84x | 1.14x | ✅ | ✅ | ✅ |
| 2 | 94 | 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm.py | level2融合 | CV | Y | / | / | / | / | / | ❌ | ❌ | ❌ |


## AscendC代码生成

### NPUKernelBench 评测子集列表

**Level 1** (29 tasks)：1-29

### AscendC基线结果

**评测环境**
- 更新时间：2026-04-08
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.0, PyTorch 2.1
- 评测范围：Level 1 (1-29)
- Agent：Asc算子生成Agent @claude-opus-4-6

**综合评测结果**
| 指标 | 结果 |
|------|------|
| **综合精度通过率** | 13/29 (44%) |
| **综合性能≥0.6x达标** | 4/29 (17%) |
| **综合性能≥0.8x达标** | 4/29 (17%) |


**Vector评测结果**
| 指标 | 结果 |
|------|------|
| **Vector精度通过率** | 13/29 (44%) |
| **Vector性能≥0.6x达标** | 4/5 (80%) |
| **Vector性能≥0.8x达标** | 4/5 (80%) |
| Vector 平均加速比 | 0.94x (注：仅5个有性能数据的任务) |


**详细结果表**

| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | PyTorch 参考延迟(ms) | 生成AscendC代码延迟(ms) | 加速比 | 最终状态 | 精度正确 | 性能 0.6x 达标 | 性能 0.8x 达标 | 备注 |
|:---|:---:|---------|:-------:|:------:|:------:|-------------:|---------------:|--------:|:-------:|:-------:|:-------:|:-------:|:---|
| 1 | 1 | GELU | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | 性能测试跳过 |
| 1 | 2 | SwiGLU | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | 精度验证失败 |
| 1 | 3 | Add | VECTOR | ✅ | ✅ | 0.032 | 0.025 | 1.28x | 成功 | ✅ | ✅ | ✅ | 有两个用例误差在容忍范围内 |
| 1 | 4 | Abs | VECTOR | ✅ | ⚠️ | \ | \ | \ | 部分成功 | ⚠️ | ❌ | ❌ | 部分通过 35/50 |
| 1 | 5 | Cumsum | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | 全量验证 47/51 |
| 1 | 6 | Histc | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | 性能测试跳过 |
| 1 | 7 | Sum | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | 性能测试跳过 |
| 1 | 8 | Sort | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | 直接使用torch实现 |
| 1 | 9 | TopK | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | 直接使用torch实现 |
| 1 | 10 | LayerNorm | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | 性能测试跳过 |
| 1 | 11 | GroupNorm | VECTOR | ✅ | ⚠️ | \ | \ | \ | 部分成功 | ⚠️ | ❌ | ❌ | 部分通过，bfloat16有精度问题 |
| 1 | 12 | Permute | VECTOR | ✅ | ✅ | 0.091 | 0.096 | 0.95x | 成功 | ✅ | ❌ | ❌ | |
| 1 | 13 | Cat | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | 性能测试跳过 |
| 1 | 14 | Split | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | 尝试导入kernel但未使用: _ext |
| 1 | 15 | Pad | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | no_kernel |
| 1 | 16 | Repeat | VECTOR | ✅ | ✅ | 0.096 | 0.098 | 0.98x | 成功 | ✅ | ✅ | ✅ | |
| 1 | 17 | AdamW | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | tilelang环境问题，UB overflow |
| 1 | 18 | Index | VECTOR | ✅ | ⚠️ | \ | \ | \ | 部分成功 | ⚠️ | ❌ | ❌ | 部分通过，CCU instruction address check error |
| 1 | 19 | IndexPut | VECTOR | ✅ | ⚠️ | \ | \ | \ | 部分成功 | ⚠️ | ❌ | ❌ | 部分通过，aicore异常，CCU instruction address check error |
| 1 | 20 | Gather | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | |
| 1 | 21 | Scatter | VECTOR | ✅ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | UB溢出 |
| 1 | 22 | Nonzero | VECTOR | ✅ | ✅ | \ | \ | \ | 成功 | ✅ | ❌ | ❌ | |
| 1 | 23 | RepeatInterleave | VECTOR | ✅ | ✅ | 0.27 | 0.271 | 0.99x | 成功 | ✅ | ✅ | ✅ | |
| 1 | 24 | EmbeddingDenseBackward | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 25 | NLLLoss | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 26 | AvgPool3d | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 27 | MaxPool3d | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 28 | Interpolate | VECTOR | ❌ | ❌ | \ | \ | \ | 失败 | ❌ | ❌ | ❌ | |
| 1 | 29 | DynamicQuant | VECTOR | ✅ | ✅ | 0.044 | 0.09 | 0.48x | 成功 | ✅ | ❌ | ❌ | |


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
