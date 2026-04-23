#include "sort_fullload.h"
#include "sort_one_core.h"
#include "sort_multi_core.h"

using namespace AscendC;
using namespace KvSort;

extern "C" __global__ __aicore__ void kv_sort_kernel(
    GM_ADDR keys, GM_ADDR values, GM_ADDR sortedKeys, GM_ADDR sortedValues,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }

    auto* gmTiling = reinterpret_cast<__gm__ SortKernelTiling*>(tiling);
    SortKernelTiling t;
    t.tilingMode = gmTiling->tilingMode;
    t.totalLength = gmTiling->totalLength;
    t.sortNum = gmTiling->sortNum;
    t.coreNum = gmTiling->coreNum;
    t.perCoreElements = gmTiling->perCoreElements;
    t.lastCoreElements = gmTiling->lastCoreElements;
    t.perCoreLoops = gmTiling->perCoreLoops;
    t.lastCoreLoops = gmTiling->lastCoreLoops;
    t.perCorePerLoopElements = gmTiling->perCorePerLoopElements;
    t.lastCorePerLoopElements = gmTiling->lastCorePerLoopElements;
    t.perCoreLastLoopElements = gmTiling->perCoreLastLoopElements;
    t.lastCoreLastLoopElements = gmTiling->lastCoreLastLoopElements;
    t.oneLoopMaxElements = gmTiling->oneLoopMaxElements;
    t.needCoreNum = gmTiling->needCoreNum;

    if (t.tilingMode == SORT_TILING_MODE_FULLLOAD) {
        TPipe pipe;
        SortFullLoad op;
        op.Init(keys, values, sortedKeys, sortedValues, &t, &pipe);
        op.Process();
    } else if (t.tilingMode == SORT_TILING_MODE_SINGLECORE) {
        TPipe pipe;
        SortOneCore op;
        op.Init(keys, values, sortedKeys, sortedValues, &t, &pipe);
        op.Process();
    } else if (t.tilingMode == SORT_TILING_MODE_MULTICORE) {
        TPipe pipe;
        SortMultiCore op;
        op.Init(keys, values, sortedKeys, sortedValues, workspace, &t, &pipe);
        op.Process();
        pipe.Destroy();
    }
}

extern "C" void kv_sort_do(
    uint32_t blockDim,
    void* stream,
    uint8_t* keys,
    uint8_t* values,
    uint8_t* sortedKeys,
    uint8_t* sortedValues,
    uint8_t* workspace,
    uint8_t* tiling)
{
    kv_sort_kernel<<<blockDim, nullptr, stream>>>(
        keys, values, sortedKeys, sortedValues, workspace, tiling);
}
