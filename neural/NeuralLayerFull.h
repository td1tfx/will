#pragma once
#include "NeuralLayer.h"
class NeuralLayerFull :
	public NeuralLayer
{
public:
	NeuralLayerFull();
	virtual ~NeuralLayerFull();

	//weight���󣬶���ȫ���Ӳ㣬�����Ǳ���Ľڵ�������������һ��Ľڵ���
	Matrix* WeightMatrix = nullptr;
	//ƫ��������ά��Ϊ����ڵ���
	Matrix* BiasVector = nullptr;
	//����ƫ�������ĸ�������������ֵΪ1��ά��Ϊ��������
	Matrix* _asBiasVector = nullptr;

protected:
	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void backPropagateDelta2() override;
public:
	void activeForwardOutput() override;
	void spreadDeltaToPrevLayer() override;
	void updateWeightBias(double learnSpeed, double lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;
};

