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

	virtual void initData2(int x1, int x2);
	virtual void resetGroupCount2();
	virtual void connetPrevlayer2();
	virtual void activeOutputValue();
	virtual void updateDelta2();
	virtual void spreadDeltaToPrevLayer();
	virtual void backPropagate(double learnSpeed, double lambda);

	virtual int saveInfo(FILE* fout);
	virtual int loadInfo(double* v, int n);
};

