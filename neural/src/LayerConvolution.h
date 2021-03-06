#pragma once
#include "Layer.h"

class LayerConvolution :
    public Layer
{
public:
    LayerConvolution();
    virtual ~LayerConvolution();

    int kernelCount = 0;
    int kernelRow, kernelCol;

    Matrix* kernelData = nullptr;
    Matrix** kernels = nullptr;
    ConvolutionType _convolutionType = cv_1toN;
    //需要一个连接方式矩阵，看起来很麻烦
    //应该是从卷积核和计算方式算出一个矩阵，这个矩阵应该是比较稀疏的
    //提供的是连接方式，卷积核，据此计算出一个大矩阵
protected:
    void init2(Option* op, const std::string& section) override {}
    void resetGroupCount2() override {}
    void connetPrevlayer2() override {}
    void activeBackward2() override {}
public:
    void activeForward() override;
    void spreadDeltaToPrevLayer() override {}
    void updateParameters() override {}
    int saveInfo(FILE* fout) override;
    int loadInfo(real* v, int n) override;

    void setSubType(ConvolutionType cv) override { _convolutionType = cv; }

};

