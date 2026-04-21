"""Tile-level TileLang design for Key-Value Sort.

Expresses the detailed algorithmic logic of each tiling mode using TileLang
primitives. The key insight is that AscendC hardware Sort only supports float32
descending order, so we negate keys before sorting and negate back afterward
to achieve ascending int32 sort.

Three modes:
- fullload:    Single UB pass — load all, sort, write.
- singlecore:  Same sort logic, but separated for pipeline clarity.
- multicore:   Block-sort per core → intra-core merge → cross-core merge tree.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

SORT_ALIGN = 32
MIN_FP32 = T.float32(-3.4e38)


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def kv_sort(totalLength, dtype="int32"):
    NUM_CORES = 20
    UB_SORT_CAPACITY = 2048
    MULTI_CORE_PER_LOOP = 512
    ONE_LOOP_MAX = 256
    MAX_MRGSORT_LIST = 4

    sortNum = T.ceildiv(totalLength, SORT_ALIGN) * SORT_ALIGN

    @T.prim_func
    def fullload(
        Keys: T.Tensor((totalLength,), dtype),
        Values: T.Tensor((totalLength,), dtype),
        SortedKeys: T.Tensor((totalLength,), dtype),
        SortedValues: T.Tensor((totalLength,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # UB allocations: key-value pair buffers in sort format
            keys_ub = T.alloc_ub((sortNum,), "int32")
            values_ub = T.alloc_ub((sortNum,), "int32")
            keys_fp32_ub = T.alloc_ub((sortNum,), "float32")
            sorted_kv_ub = T.alloc_ub((sortNum * 2,), "float32")  # sort format: interleaved k,v
            out_keys_ub = T.alloc_ub((sortNum,), "float32")
            out_vals_ub = T.alloc_ub((sortNum,), "uint32")
            temp_ub = T.alloc_ub((sortNum * 2,), "float32")

            with T.Scope("V"):
                # Step 1: CopyIn — load keys and values from GM to UB
                T.copy(Keys[0:totalLength], keys_ub[0:totalLength])
                T.copy(Values[0:totalLength], values_ub[0:totalLength])

                # Step 2: Cast int32 → float32 (Sort instruction requires float)
                T.tile.cast(keys_fp32_ub, keys_ub, mode="CAST_ROUND", count=sortNum)

                # Step 3: Negate keys (Sort is descending; negating gives ascending)
                T.tile.mul(keys_fp32_ub, keys_fp32_ub, T.float32(-1.0))

                # Step 4: Pad tail elements (not aligned to 32) with MIN_FP32
                # so they sink to the end of descending sort and don't interfere
                # (conceptual — actual mask-based Duplicate in AscendC)

                # Step 5: Concat — rearrange into Sort instruction's input layout
                # (groups of 32 elements packed for hardware Sort unit)
                # T.tile.concat(concat_ub, keys_fp32_ub, temp_ub, sortNum // SORT_ALIGN)

                # Step 6: Hardware Sort — key-value descending sort
                # Sort<float, true>(sorted_kv_ub, concat_ub, values_ub, temp_ub, sortNum // 32)
                # This sorts keys (as float32) in descending order,
                # carrying values (reinterpreted as uint32) along.

                # Step 7: Extract — separate sorted keys and values from sort format
                # T.tile.extract(out_keys_ub, out_vals_ub, sorted_kv_ub, sortNum // 32)

                # Step 8: Negate keys back to restore original values
                T.tile.mul(out_keys_ub, out_keys_ub, T.float32(-1.0))

                # Step 9: Cast float32 → int32
                # T.tile.cast(out_keys_int_ub, out_keys_ub, mode="CAST_ROUND", count=sortNum)

                # Step 10: CopyOut — write sorted results to GM
                # T.copy(out_keys_int_ub[0:totalLength], SortedKeys[0:totalLength])
                # T.copy(out_vals_ub[0:totalLength], SortedValues[0:totalLength])

                _ = sorted_kv_ub
                _ = out_vals_ub
                _ = temp_ub
                _ = SortedKeys
                _ = SortedValues

    @T.prim_func
    def singlecore(
        Keys: T.Tensor((totalLength,), dtype),
        Values: T.Tensor((totalLength,), dtype),
        SortedKeys: T.Tensor((totalLength,), dtype),
        SortedValues: T.Tensor((totalLength,), dtype),
    ):
        """Identical sort logic to fullload.
        Separated because the AscendC implementation uses different output routing
        (workspace vs direct GM) to interface with downstream phases.
        """
        with T.Kernel(1, is_npu=True) as (cid, vid):
            keys_ub = T.alloc_ub((sortNum,), "int32")
            values_ub = T.alloc_ub((sortNum,), "int32")
            keys_fp32_ub = T.alloc_ub((sortNum,), "float32")
            temp_ub = T.alloc_ub((sortNum * 2,), "float32")

            with T.Scope("V"):
                T.copy(Keys[0:totalLength], keys_ub[0:totalLength])
                T.copy(Values[0:totalLength], values_ub[0:totalLength])

                T.tile.cast(keys_fp32_ub, keys_ub, mode="CAST_ROUND", count=sortNum)
                T.tile.mul(keys_fp32_ub, keys_fp32_ub, T.float32(-1.0))

                # Pad + Concat + Sort + Extract + Negate + Cast + CopyOut
                # (same as fullload)
                _ = temp_ub
                _ = SortedKeys
                _ = SortedValues

    needCores = T.min(NUM_CORES, T.ceildiv(totalLength, MULTI_CORE_PER_LOOP))
    perCoreElements = T.ceildiv(totalLength, needCores)
    loopElements = T.min(MULTI_CORE_PER_LOOP, perCoreElements)

    @T.prim_func
    def multicore(
        Keys: T.Tensor((totalLength,), dtype),
        Values: T.Tensor((totalLength,), dtype),
        SortedKeys: T.Tensor((totalLength,), dtype),
        SortedValues: T.Tensor((totalLength,), dtype),
    ):
        with T.Kernel(needCores, is_npu=True) as (cid, vid):
            # Per-core UB: holds one sort block (loopElements key-value pairs)
            blk_keys_ub = T.alloc_ub((loopElements,), "int32")
            blk_vals_ub = T.alloc_ub((loopElements,), "int32")
            blk_fp32_ub = T.alloc_ub((loopElements,), "float32")
            sort_temp_ub = T.alloc_ub((loopElements * 2,), "float32")
            sorted_block_ub = T.alloc_ub((loopElements * 2,), "float32")  # sort format

            # Merge buffers for MrgSort (up to 4 input lists)
            mrg_in_ub = T.alloc_ub((MAX_MRGSORT_LIST, ONE_LOOP_MAX * 2), "float32")
            mrg_out_ub = T.alloc_ub((ONE_LOOP_MAX * MAX_MRGSORT_LIST * 2,), "float32")

            with T.Scope("V"):
                coreStart = cid * perCoreElements
                coreEnd = T.min(coreStart + perCoreElements, totalLength)
                coreLen = coreEnd - coreStart
                numLoops = T.ceildiv(coreLen, loopElements)

                # ============================================================
                # PHASE 1: VBS — each core sorts its local blocks
                # ============================================================
                for loop in T.serial(numLoops):
                    blockStart = coreStart + loop * loopElements
                    blockEnd = T.min(blockStart + loopElements, coreEnd)
                    blockLen = blockEnd - blockStart
                    sn = T.ceildiv(blockLen, SORT_ALIGN) * SORT_ALIGN

                    # Load one block of keys + values into UB
                    T.copy(Keys[blockStart:blockEnd], blk_keys_ub[0:blockLen])
                    T.copy(Values[blockStart:blockEnd], blk_vals_ub[0:blockLen])

                    # int32 → float32, negate
                    T.tile.cast(blk_fp32_ub, blk_keys_ub, mode="CAST_ROUND", count=sn)
                    T.tile.mul(blk_fp32_ub, blk_fp32_ub, T.float32(-1.0))

                    # Pad tail with MIN_FP32
                    # Concat → Sort<float,true> → result in sorted_block_ub (sort format)
                    # Write sorted block to workspace GM (ping buffer)
                    # workspace[cid * GetSortLen(perCoreElements) + GetSortLen(loop * loopElements)]

                    _ = sort_temp_ub
                    _ = sorted_block_ub
                    _ = sn

                # Intra-core merge: merge numLoops sorted blocks into one segment
                # Using MrgSort hardware instruction (≤4-way merge per round)
                # Workspace ping-pong between two GM buffers
                # listNum = numLoops
                # while listNum > 1:
                #     merge groups of min(4, listNum) lists
                #     swap ping/pong workspace
                #     listNum = ceil(listNum / 4)
                _ = mrg_in_ub
                _ = mrg_out_ub

                # SyncAll — all cores have one sorted segment in workspace

                # ============================================================
                # PHASE 2: VMS — cross-core merge tree
                # ============================================================
                # Now needCores sorted segments exist in workspace.
                # Each round: ceil(listNum/4) cores each merge 4 segments.
                # Active cores decrease by 4x per round.
                # SyncAll between rounds.
                # Continues until listNum ≤ 4.
                #
                # listNum = needCores
                # while listNum > 4:
                #     activeCores = ceil(listNum / 4)
                #     if cid < activeCores:
                #         merge 4 segments from workspace[srcWs] → workspace[dstWs]
                #     swap srcWs/dstWs
                #     listNum = activeCores
                #     SyncAll

                # ============================================================
                # PHASE 3: SortOut — final merge on core 0
                # ============================================================
                # Core 0 merges remaining ≤4 segments:
                #   for each MrgSort output chunk:
                #     CopyIn from workspace
                #     MrgSort ≤4-way
                #     Extract: separate key (float32) and value (uint32)
                #     Negate keys, Cast float32 → int32
                #     CopyOut to SortedKeys and SortedValues GM

                _ = SortedKeys
                _ = SortedValues

    if totalLength <= UB_SORT_CAPACITY:
        return fullload
    if totalLength <= UB_SORT_CAPACITY * 2:
        return singlecore
    return multicore
