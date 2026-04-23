#include <algorithm>
#include <cstring>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "sort_tiling.h"

extern "C" void kv_sort_do(
    uint32_t blockDim,
    void* stream,
    uint8_t* keys,
    uint8_t* values,
    uint8_t* sortedKeys,
    uint8_t* sortedValues,
    uint8_t* workspace,
    uint8_t* tiling);

namespace kv_sort_ext {

constexpr int32_t NUM_CORES = 20;
constexpr int64_t UB_SORT_CAPACITY = 2048;
constexpr int64_t MULTI_CORE_PER_LOOP = 512;
constexpr int64_t ONE_LOOP_MAX = 256;

static void computeTiling(SortKernelTiling* t, int32_t totalLength)
{
    std::memset(t, 0, sizeof(SortKernelTiling));
    t->totalLength = totalLength;

    int64_t aligned = ((int64_t)totalLength * 4 + 31) / 32 * 32 / 4;
    int64_t sortNum = ((aligned + 31) / 32) * 32;
    t->sortNum = static_cast<int32_t>(sortNum);

    if (totalLength <= (int32_t)UB_SORT_CAPACITY) {
        t->tilingMode = SORT_TILING_MODE_FULLLOAD;
        t->coreNum = 1;
        t->needCoreNum = 1;
        return;
    }

    if (totalLength <= (int32_t)(UB_SORT_CAPACITY * 2)) {
        t->tilingMode = SORT_TILING_MODE_SINGLECORE;
        t->coreNum = 1;
        t->needCoreNum = 1;
        return;
    }

    t->tilingMode = SORT_TILING_MODE_MULTICORE;

    int32_t perLoop = static_cast<int32_t>(MULTI_CORE_PER_LOOP);
    int32_t needCores = std::min<int32_t>(NUM_CORES, (totalLength + perLoop - 1) / perLoop);
    if (needCores < 2) needCores = 2;

    t->coreNum = needCores;
    t->needCoreNum = needCores;

    int32_t perCore = (totalLength + needCores - 1) / needCores;
    perCore = ((perCore + 31) / 32) * 32;
    int32_t lastCore = totalLength - perCore * (needCores - 1);
    if (lastCore <= 0) {
        perCore = totalLength / needCores;
        perCore = ((perCore + 31) / 32) * 32;
        lastCore = totalLength - perCore * (needCores - 1);
    }

    t->perCoreElements = perCore;
    t->lastCoreElements = lastCore;

    int32_t loopElem = std::min<int32_t>(perLoop, perCore);
    loopElem = ((loopElem + 31) / 32) * 32;

    t->perCorePerLoopElements = loopElem;
    t->perCoreLoops = (perCore + loopElem - 1) / loopElem;
    t->perCoreLastLoopElements = perCore - loopElem * (t->perCoreLoops - 1);

    t->lastCorePerLoopElements = loopElem;
    t->lastCoreLoops = (lastCore + loopElem - 1) / loopElem;
    t->lastCoreLastLoopElements = lastCore - loopElem * (t->lastCoreLoops - 1);

    t->oneLoopMaxElements = static_cast<int32_t>(ONE_LOOP_MAX);
}

pybind11::tuple run_kv_sort(
    const at::Tensor& keys,
    const at::Tensor& values,
    int32_t tilingModeOverride)
{
    TORCH_CHECK(keys.dim() == 1, "keys must be 1D");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");
    TORCH_CHECK(keys.size(0) == values.size(0), "keys and values must have same length");
    TORCH_CHECK(keys.scalar_type() == at::kInt, "keys must be int32");
    TORCH_CHECK(values.scalar_type() == at::kInt, "values must be int32");
    TORCH_CHECK(keys.is_contiguous() && values.is_contiguous(), "inputs must be contiguous");

    int32_t totalLength = static_cast<int32_t>(keys.size(0));

    at::Tensor sortedKeys = at::empty_like(keys);
    at::Tensor sortedValues = at::empty_like(values);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(SortKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto* tiling = reinterpret_cast<SortKernelTiling*>(tilingCpu.data_ptr());
    computeTiling(tiling, totalLength);

    if (tilingModeOverride >= 0) {
        tiling->tilingMode = tilingModeOverride;
        if (tilingModeOverride == SORT_TILING_MODE_MULTICORE && tiling->coreNum < 2) {
            tiling->coreNum = 2;
            tiling->needCoreNum = 2;
            int32_t perCore = (totalLength + 1) / 2;
            perCore = ((perCore + 31) / 32) * 32;
            tiling->perCoreElements = perCore;
            tiling->lastCoreElements = totalLength - perCore;
            if (tiling->lastCoreElements <= 0) {
                tiling->perCoreElements = totalLength / 2;
                tiling->perCoreElements = ((tiling->perCoreElements + 31) / 32) * 32;
                tiling->lastCoreElements = totalLength - tiling->perCoreElements;
            }
            int32_t loopElem = std::min<int32_t>((int32_t)MULTI_CORE_PER_LOOP, tiling->perCoreElements);
            loopElem = ((loopElem + 31) / 32) * 32;
            tiling->perCorePerLoopElements = loopElem;
            tiling->perCoreLoops = (tiling->perCoreElements + loopElem - 1) / loopElem;
            tiling->perCoreLastLoopElements = tiling->perCoreElements - loopElem * (tiling->perCoreLoops - 1);
            int32_t lastLoopElem = std::min<int32_t>(loopElem, tiling->lastCoreElements);
            tiling->lastCorePerLoopElements = lastLoopElem;
            tiling->lastCoreLoops = (tiling->lastCoreElements + lastLoopElem - 1) / lastLoopElem;
            tiling->lastCoreLastLoopElements = tiling->lastCoreElements - lastLoopElem * (tiling->lastCoreLoops - 1);
            tiling->oneLoopMaxElements = static_cast<int32_t>(ONE_LOOP_MAX);
        }
    }

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    int64_t wsBytes = 0;
    if (tiling->tilingMode == SORT_TILING_MODE_MULTICORE) {
        int64_t perCoreSortLen = (int64_t)tiling->perCoreElements * 2;
        int64_t oneBufFloats = (int64_t)tiling->needCoreNum * perCoreSortLen;
        wsBytes = oneBufFloats * sizeof(float) * 2 + 4096;
    }
    at::Tensor workspace = at::empty({wsBytes}, keys.options().dtype(at::kByte));

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    kv_sort_do(
        tiling->coreNum,
        aclStream,
        static_cast<uint8_t*>(const_cast<void*>(keys.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(values.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(sortedKeys.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(sortedValues.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(workspace.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(tilingNpu.storage().data())));

    return pybind11::make_tuple(sortedKeys, sortedValues, tiling->tilingMode);
}

}  // namespace kv_sort_ext

PYBIND11_MODULE(_kv_sort_ext, m)
{
    m.doc() = "Key-Value Sort AscendC kernel extension";
    m.def("run_kv_sort", &kv_sort_ext::run_kv_sort,
          "Run key-value sort on NPU",
          pybind11::arg("keys"),
          pybind11::arg("values"),
          pybind11::arg("tiling_mode_override") = -1);
}
