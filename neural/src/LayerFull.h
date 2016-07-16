#pragma once
#include "Layer.h"

class LayerFull :
    public Layer
{
public:
    LayerFull();
    virtual ~LayerFull();

    //weight���󣬶���ȫ���Ӳ㣬�����Ǳ���Ľڵ�������������һ��Ľڵ���
    Matrix* WeightMatrix = nullptr;
    //ƫ��������ά��Ϊ����ڵ���
    Matrix* BiasVector = nullptr;
    //����ƫ�������ĸ�������������ֵΪ1��ά��Ϊ��������
    Matrix* asBiasVector = nullptr;

protected:
    void initData2(LayerInitInfo* info) override;
    void resetGroupCount2() override;
    void connetPrevlayer2() override;
    void activeBackward2() override;
public:
    void activeForward() override;
    void spreadDeltaToPrevLayer() override;
    void updateParameters(real learnSpeed, real lambda) override;
    int saveInfo(FILE* fout) override;
    int loadInfo(real* v, int n) override;
};
