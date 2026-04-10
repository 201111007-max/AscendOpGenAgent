#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "rms_norm_tiling.h"

template <typename dataType>
class RmsNormSplitDKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * tiling_.N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            pipe_->InitBuffer(xInQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(gammaInQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(accumBuf_, kTileFloatBytes);
            pipe_->InitBuffer(reduceBuf_, kBlockN * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(tempBuf_, kTileFloatBytes);
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, kTileFloatBytes);
                pipe_->InitBuffer(gammaCastBuf_, kTileFloatBytes);
                pipe_->InitBuffer(yCastBuf_, kTileFloatBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }

                for (int row = 0; row < subBlockRows_; ++row) {
                    const int rowIdx = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    static constexpr int kBlockN = 1024;
    static constexpr uint32_t kTileFloatBytes = kBlockN * sizeof(float);

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline int32_t NumTiles() const
    {
        return (tiling_.N + kBlockN - 1) / kBlockN;
    }

    __aicore__ inline int32_t GetValidN(int32_t colBase) const
    {
        return (colBase + kBlockN <= tiling_.N) ? kBlockN : (tiling_.N - colBase);
    }

    template <typename T>
    __aicore__ inline uint32_t ValidBytes(int32_t validN) const
    {
        return static_cast<uint32_t>(validN * static_cast<int32_t>(sizeof(T)));
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbTile(
        AscendC::LocalTensor<T> &dst,
        AscendC::GlobalTensor<T> src,
        int32_t validN)
    {
        const uint32_t validBytes = ValidBytes<T>(validN);
        const uint32_t padElems = static_cast<uint32_t>(kBlockN - validN);
        AscendC::DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyUbToGmTile(
        AscendC::GlobalTensor<T> dst,
        AscendC::LocalTensor<T> &src,
        int32_t validN)
    {
        AscendC::DataCopyExtParams copyParams{1, ValidBytes<T>(validN), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void PrepareInputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
        }
    }

    __aicore__ inline void FinalizeOutputTensor(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(out, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        accumLocal_ = accumBuf_.Get<float>();
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        tempLocal_ = tempBuf_.Get<float>();

        AscendC::Duplicate(accumLocal_, 0.0f, kBlockN);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            xInQueue_.AllocTensor<dataType>(xInLocal_);
            CopyGmToUbTile(xInLocal_, xGM_[rowIdx * tiling_.N + colBase], validN);
            xInQueue_.EnQue(xInLocal_);

            xInQueue_.DeQue<dataType>(xInLocal_);
            PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, kBlockN);
            AscendC::Mul(tempLocal_, xLocal_, xLocal_, kBlockN);
            AscendC::Add(accumLocal_, accumLocal_, tempLocal_, kBlockN);
            xInQueue_.FreeTensor(xInLocal_);
        }

        AscendC::ReduceSum<float>(sumLocal_, accumLocal_, reduceLocal_, kBlockN);
        float invRms = sumLocal_.GetValue(0) * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal_, invRms, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);
        invRms = sumLocal_.GetValue(0);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            xInQueue_.AllocTensor<dataType>(xInLocal_);
            gammaInQueue_.AllocTensor<dataType>(gammaInLocal_);
            CopyGmToUbTile(xInLocal_, xGM_[rowIdx * tiling_.N + colBase], validN);
            CopyGmToUbTile(gammaInLocal_, gammaGM_[colBase], validN);
            xInQueue_.EnQue(xInLocal_);
            gammaInQueue_.EnQue(gammaInLocal_);

            yOutQueue_.AllocTensor<dataType>(yOutLocal_);
            xInQueue_.DeQue<dataType>(xInLocal_);
            gammaInQueue_.DeQue<dataType>(gammaInLocal_);
            PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, kBlockN);
            PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, kBlockN);
            PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_);
            AscendC::Muls(yLocal_, xLocal_, invRms, kBlockN);
            AscendC::Mul(yLocal_, yLocal_, gammaLocal_, kBlockN);
            FinalizeOutputTensor(yOutLocal_, yLocal_, kBlockN);
            xInQueue_.FreeTensor(xInLocal_);
            gammaInQueue_.FreeTensor(gammaInLocal_);
            yOutQueue_.EnQue(yOutLocal_);

            yOutQueue_.DeQue<dataType>(yOutLocal_);
            CopyUbToGmTile(yGM_[rowIdx * tiling_.N + colBase], yOutLocal_, validN);
            yOutQueue_.FreeTensor(yOutLocal_);
        }
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gammaInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> accumLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
    AscendC::LocalTensor<float> tempLocal_;
};
