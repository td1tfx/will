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

