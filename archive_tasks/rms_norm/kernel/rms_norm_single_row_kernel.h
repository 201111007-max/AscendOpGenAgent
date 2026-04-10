#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "rms_norm_tiling.h"

template <typename dataType>
class RmsNormSingleRowKernel {
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
            pipe_->InitBuffer(gammaBuf_, RowBytes<dataType>());
            pipe_->InitBuffer(xInQueue_, 1, RowBytes<dataType>());
            pipe_->InitBuffer(yOutQueue_, 1, RowBytes<dataType>());
            pipe_->InitBuffer(reduceBuf_, tiling_.N * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, tiling_.N * sizeof(float));
                pipe_->InitBuffer(gammaCastBuf_, tiling_.N * sizeof(float));
                pipe_->InitBuffer(yCastBuf_, tiling_.N * sizeof(float));
            }

            gammaInLocal_ = gammaBuf_.Get<dataType>();
            CopyGmToUbRow(gammaInLocal_, gammaGM_);
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::PipeBarrier<PIPE_ALL>();
            PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, tiling_.N);
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
    template <typename T>
    __aicore__ inline uint32_t RowBytes() const
    {
        return static_cast<uint32_t>(tiling_.N * sizeof(T));
    }

    template <typename T>
    __aicore__ inline void CopyUbToGmRow(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> &src)
    {
        AscendC::DataCopyExtParams copyParams{1, RowBytes<T>(), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbRow(AscendC::LocalTensor<T> &dst, AscendC::GlobalTensor<T> src)
    {
        AscendC::DataCopyExtParams copyParams{1, RowBytes<T>(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
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
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Duplicate(out, static_cast<dataType>(0), count);
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
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();

        CopyGmToUbRow(xInLocal_, xGM_[rowIdx * tiling_.N]);
        xInQueue_.EnQue(xInLocal_);

        xInQueue_.DeQue<dataType>(xInLocal_);
        PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, tiling_.N);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_, tiling_.N);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, tiling_.N);
        AscendC::ReduceSum<float>(sumLocal_, yLocal_, reduceLocal_, tiling_.N);

        float meanSq = sumLocal_.GetValue(0) * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal_, meanSq, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);

        float invRms = sumLocal_.GetValue(0);
        AscendC::Muls(yLocal_, xLocal_, invRms, tiling_.N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, tiling_.N);
        FinalizeOutputTensor(yOutLocal_, yLocal_, tiling_.N);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);

        yOutQueue_.DeQue<dataType>(yOutLocal_);
        CopyUbToGmRow(yGM_[rowIdx * tiling_.N], yOutLocal_);
        yOutQueue_.FreeTensor(yOutLocal_);
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};
