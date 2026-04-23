#ifndef SORT_MRGSORT_OUT_H
#define SORT_MRGSORT_OUT_H

#include "sort_mrgsort.h"

namespace KvSort {
using namespace AscendC;

class KvMrgSortOut
{
public:
    __aicore__ inline KvMrgSortOut() {};
    __aicore__ inline void Init(MrgSortParam* param, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput);
    __aicore__ inline void SetOutput(
        GlobalTensor<int32_t>& gmKeyOut, GlobalTensor<int32_t>& gmValOut,
        LocalTensor<float>& ubOutput1, LocalTensor<float>& ubOutput2);
    __aicore__ inline void SetBuffer(LocalTensor<float>& tempBuffer);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgSortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void ExtractAndRestore();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    MrgSortParam* param = nullptr;
    GlobalTensor<float> gmInputs[4];
    GlobalTensor<int32_t> gmKeyOut;
    GlobalTensor<int32_t> gmValOut;

    LocalTensor<float> ubInputs[4];
    LocalTensor<float> tempBuf;
    LocalTensor<float> ubKeyOut;
    LocalTensor<uint32_t> ubValOut;
    LocalTensor<int32_t> ubKeyOutInt;
    LocalTensor<int32_t> ubValOutInt;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    uint16_t validBitTail;
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];
    LocalTensor<float> tmpUbInputs[4];
};

__aicore__ inline void KvMrgSortOut::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

__aicore__ inline void KvMrgSortOut::SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

__aicore__ inline void KvMrgSortOut::SetOutput(
    GlobalTensor<int32_t>& gmKeyOut, GlobalTensor<int32_t>& gmValOut,
    LocalTensor<float>& ubOutput1, LocalTensor<float>& ubOutput2)
{
    this->gmKeyOut = gmKeyOut;
    this->ubKeyOut = ubOutput1;
    this->ubKeyOutInt = ubOutput1.ReinterpretCast<int32_t>();

    this->gmValOut = gmValOut;
    this->ubValOut = ubOutput2.ReinterpretCast<uint32_t>();
    this->ubValOutInt = ubOutput2.ReinterpretCast<int32_t>();
}

__aicore__ inline void KvMrgSortOut::SetBuffer(LocalTensor<float>& tempBuffer)
{
    this->tempBuf = tempBuffer;
}

__aicore__ inline void KvMrgSortOut::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

__aicore__ inline void KvMrgSortOut::CopyIn()
{
    this->remainListNum = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = MinVal(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]],
                     AlignUp(GetSortLen<float>(lengths[i]), sizeof(float)));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

__aicore__ inline void KvMrgSortOut::MrgSortCompute()
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (this->remainListNum == MERGE_LIST_TWO) {
        MrgSortSrcList sortList = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuf, sortList, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        MrgSortSrcList sortList = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuf, sortList, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        MrgSortSrcList sortList = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->tempBuf, sortList, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        DataCopy(this->tempBuf, this->tmpUbInputs[0],
                 AlignUp(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        listSortedNums[0] = elementCountListTail[0];
    }
}

__aicore__ inline void KvMrgSortOut::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

__aicore__ inline void KvMrgSortOut::ExtractAndRestore()
{
    AscendC::Extract(this->ubKeyOut, this->ubValOut, this->tempBuf,
                     CeilDiv(curLoopSortedNum, ONE_REPEAT_SORT_NUM));
    Muls(this->ubKeyOut, this->ubKeyOut, (float)-1, AlignUp(curLoopSortedNum, sizeof(float)));
    Cast(this->ubKeyOutInt, this->ubKeyOut, RoundMode::CAST_ROUND, AlignUp(curLoopSortedNum, sizeof(float)));
}

__aicore__ inline void KvMrgSortOut::CopyOut()
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(this->gmKeyOut[outOffset], this->ubKeyOutInt, intriParams);
    DataCopyPad(this->gmValOut[outOffset], this->ubValOutInt, intriParams);
    outOffset += curLoopSortedNum;
}

__aicore__ inline void KvMrgSortOut::Init(MrgSortParam* param, TPipe* tPipe)
{
    this->param = param;
    this->allRemainElements = 0;
    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
}

__aicore__ inline void KvMrgSortOut::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgSortCompute();
        UpdateSortInfo();
        ExtractAndRestore();
        CopyOut();
    }
    ClearCache();
}

}  // namespace KvSort
#endif
