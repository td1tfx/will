#pragma once
#include "Layer.h"

class LayerFull :
    public Layer
{
public:
    LayerFull();
    virtual ~LayerFull();

    //weight矩阵，对于全连接层，行数是本层的节点数，列数是上一层的节点数
    Matrix* WeightMatrix = nullptr;
    //偏移向量，维度为本层节点数
    Matrix* BiasVector = nullptr;
    //更新偏移向量的辅助向量，所有值为1，维度为数据组数
    Matrix* asBiasVector = nullptr;

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
};

