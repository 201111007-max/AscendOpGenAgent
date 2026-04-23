#ifndef SORT_COMMON_H
#define SORT_COMMON_H

#include "kernel_operator.h"

namespace KvSort {
using namespace AscendC;

constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t DST_BLK_STRIDE = 1;
constexpr int64_t DST_REP_STRIDE = 8;
constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;
constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;
constexpr int64_t MAX_MRGSORT_LIST = 4;
constexpr int64_t WORK_GM_NUM = 2;

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) return 0;
    return (a + b - 1) / b;
}

__aicore__ inline int64_t AlignUp(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) return 0;
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignUpBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T MinVal(T a, T b) { return a > b ? b : a; }

template <typename T>
__aicore__ inline T MaxVal(T a, T b) { return a < b ? b : a; }

}  // namespace KvSort
#endif
