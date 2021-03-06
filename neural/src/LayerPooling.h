#pragma once
#include "Layer.h"

class LayerPooling :
    public Layer
{
public:
    LayerPooling();
    virtual ~LayerPooling();

    int* recordPos = nullptr;   //记录最大值的位置

    PoolingType _resampleType = pl_Max;
    real Weight, Bias;
    //所有值为1
    Matrix* _asBiasMatrix = nullptr;

    int window_w, window_h;  //pooling窗口尺寸
    int stride_w, stride_h;  //pooling步长

protected:
    void init2(Option* op, const std::string& section) override;
    void resetGroupCount2() override;
    void connetPrevlayer2() override;
    void activeBackward2() override;
public:
    void activeForward() override;
    void spreadDeltaToPrevLayer() override;
    void updateParameters() override;
    int saveInfo(FILE* fout) override;
    int loadInfo(real* v, int n) override;

    void setSubType(PoolingType re) override { _resampleType = re; }
};

