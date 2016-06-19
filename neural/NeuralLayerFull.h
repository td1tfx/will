#pragma once
#include "NeuralLayer.h"
class NeuralLayerFull :
	public NeuralLayer
{
public:
	NeuralLayerFull();
	virtual ~NeuralLayerFull();

	//weight���󣬶���ȫ���Ӳ㣬�����Ǳ���Ľڵ�������������һ��Ľڵ���
	d_matrix* WeightMatrix = nullptr;
	//ƫ��������ά��Ϊ����ڵ���
	d_matrix* BiasVector = nullptr;
	//����ƫ�������ĸ�������������ֵΪ1��ά��Ϊ��������
	d_matrix* _asBiasVector = nullptr;

	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void activeOutputValue() override;
	void updateDelta2() override;
	void spreadDeltaToPrevLayer() override;
	void backPropagate(double learnSpeed, double lambda) override;

	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;
};

