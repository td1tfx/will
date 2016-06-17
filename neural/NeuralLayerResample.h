#pragma once
#include "NeuralLayer.h"
class NeuralLayerResample :
	public NeuralLayer
{
public:

	//在处理图像模式的时候，上一层output的向量在这里被转为矩阵
	d_matrix* outputMatrix_image = nullptr;
	int scale_m, scale_n;

	NeuralLayerResample();
	virtual ~NeuralLayerResample();

	virtual void initData2(int x1, int x2);
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2();
	virtual void activeOutputValue();
	virtual void updateDelta() {}
	virtual void spreadDeltaToPrevLayer() {}
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(double* v, int n) { return 0; }
};

