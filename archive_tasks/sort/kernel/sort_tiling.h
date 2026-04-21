#ifndef SORT_TILING_H
#define SORT_TILING_H

#include <cstdint>

constexpr int32_t SORT_TILING_MODE_FULLLOAD = 0;
constexpr int32_t SORT_TILING_MODE_SINGLECORE = 1;
constexpr int32_t SORT_TILING_MODE_MULTICORE = 2;

constexpr int32_t SORT_MAX_CORES = 20;

struct SortKernelTiling {
    int32_t tilingMode;
    int32_t totalLength;
    int32_t sortNum;
    int32_t coreNum;

    int32_t perCoreElements;
    int32_t lastCoreElements;
    int32_t perCoreLoops;
    int32_t lastCoreLoops;
    int32_t perCorePerLoopElements;
    int32_t lastCorePerLoopElements;
    int32_t perCoreLastLoopElements;
    int32_t lastCoreLastLoopElements;

    int32_t oneLoopMaxElements;

    int32_t needCoreNum;
};

#endif
