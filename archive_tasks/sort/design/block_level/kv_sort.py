"""Block-level TileLang design for Key-Value Sort.

Three tiling strategies based on data size:
- `fullload`: data fits entirely in UB, single-core single-pass sort.
- `singlecore`: data fits in one core's UB for sort, but needs workspace for
  multi-phase pipeline (sort → output).
- `multicore`: data must be partitioned across cores; each core sorts locally,
  then a multi-level merge tree produces the globally sorted result.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def kv_sort(totalLength, dtype="int32"):
    NUM_CORES = 20
    UB_SORT_CAPACITY = 2048
    MULTI_CORE_PER_LOOP = 512
    MAX_MRGSORT_LIST = 4

    @T.prim_func
    def fullload(
        Keys: T.Tensor((totalLength,), dtype),
        Values: T.Tensor((totalLength,), dtype),
        SortedKeys: T.Tensor((totalLength,), dtype),
        SortedValues: T.Tensor((totalLength,), dtype),
    ):
        """FullLoad mode: entire key/value arrays fit in UB.

        Block-level view:
          - Single core (core 0) loads all data into UB.
          - One hardware Sort pass produces the sorted result.
          - Directly write back to GM outputs.
        """
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                # TODO(tile-level):
                # - Load keys[0:totalLength] and values[0:totalLength] into UB
                # - Cast keys int32 → float32, negate for ascending via descending Sort
                # - Pad tail to 32-element alignment with MIN_FP32
                # - Concat → Sort<float,true> → Extract
                # - Negate + Cast back to int32
                # - Write sorted keys and values to GM
                _ = Keys
                _ = Values
                _ = SortedKeys
                _ = SortedValues

    @T.prim_func
    def singlecore(
        Keys: T.Tensor((totalLength,), dtype),
        Values: T.Tensor((totalLength,), dtype),
        SortedKeys: T.Tensor((totalLength,), dtype),
        SortedValues: T.Tensor((totalLength,), dtype),
    ):
        """SingleCore mode: data fits in one core's UB for sorting.

        Block-level view:
          - Core 0 performs the sort; all other cores idle during sort phase.
          - Sort result written to output GM directly.
          - Identical to FullLoad in block structure, separated for tiling clarity.
        """
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                # TODO(tile-level):
                # - Same as fullload: load, cast, negate, sort, extract, restore, write
                _ = Keys
                _ = Values
                _ = SortedKeys
                _ = SortedValues

    needCores = T.min(NUM_CORES, T.ceildiv(totalLength, MULTI_CORE_PER_LOOP))
    perCoreElements = T.ceildiv(totalLength, needCores)
    lastCoreElements = totalLength - perCoreElements * (needCores - 1)

    @T.prim_func
    def multicore(
        Keys: T.Tensor((totalLength,), dtype),
        Values: T.Tensor((totalLength,), dtype),
        SortedKeys: T.Tensor((totalLength,), dtype),
        SortedValues: T.Tensor((totalLength,), dtype),
    ):
        """MultiCore mode: data partitioned across multiple cores.

        Block-level view — three phases:

        Phase 1 (VBS): Vector Block Sort — all cores in parallel
          - Each core loads its partition of keys/values.
          - Splits into UB-sized blocks, sorts each block with hardware Sort.
          - Merges blocks within the core using MrgSort (≤4-way merge).
          - Writes one sorted segment per core to workspace GM (ping-pong buffer).

        Phase 2 (VMS): Vector Merge Sort — progressively fewer cores
          - Treats each core's sorted segment as one list.
          - Each round: groups of ≤4 lists are merged by one core.
          - Active cores decrease by 4× each round.
          - Continues until ≤4 lists remain.
          - SyncAll between rounds.

        Phase 3 (SortOut): Final merge — single core
          - Core 0 merges the remaining ≤4 lists.
          - Extract separates key/value from sort format.
          - Negates keys back and casts float32 → int32.
          - Writes final sorted_keys and sorted_values to output GM.
        """
        with T.Kernel(needCores, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                # --- Phase 1: VBS (each core sorts its partition) ---
                # Each core handles keys[cid*perCoreElements : (cid+1)*perCoreElements]
                # Sub-blocks sorted via Sort instruction, then merged within core

                # --- Phase 2: VMS (cross-core merge tree) ---
                # Round by round, groups of 4 sorted segments are merged
                # workspace ping-pong between two GM buffers
                # SyncAll after each round

                # --- Phase 3: SortOut (final merge on core 0) ---
                # ≤4 remaining segments merged, Extract + restore, write output

                _ = Keys
                _ = Values
                _ = SortedKeys
                _ = SortedValues
                _ = perCoreElements
                _ = lastCoreElements
                _ = MAX_MRGSORT_LIST

    if totalLength <= UB_SORT_CAPACITY:
        return fullload
    if totalLength <= UB_SORT_CAPACITY * 2:
        return singlecore
    return multicore
