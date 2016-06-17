#pragma once
#include "NeuralLayer.h"
class NeuralLayerResample :
	public NeuralLayer
{
public:

	//�ڴ���ͼ��ģʽ��ʱ����һ��output�����������ﱻתΪ����
	d_matrix* outputMatrix_image = nullptr;
	int region_m = 2, region_n = 2;

	NeuralLayerResample();
	virtual ~NeuralLayerResample();

	virtual void initData2(int x1, int x2);
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2();
	virtual void updateDelta2() {}

	virtual void activeOutputValue();
	virtual void spreadDeltaToPrevLayer() {}
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(double* v, int n) { return 0; }
};

