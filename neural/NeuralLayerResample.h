#pragma once
#include "NeuralLayer.h"
class NeuralLayerResample :
	public NeuralLayer
{
public:

	//在处理图像模式的时候，上一层output的向量在这里被转为矩阵
	int region_m = 2, region_n = 2;

	NeuralLayerResample();
	virtual ~NeuralLayerResample();

	ResampleType _resample = re_Findmax;

	virtual void initData2(int x1, int x2);
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2();
	virtual void updateDelta2() {}

	virtual void activeOutputValue();
	virtual void spreadDeltaToPrevLayer();
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout);
	virtual int loadInfo(double* v, int n);
};

