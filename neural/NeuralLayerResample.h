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

	int* maxPos = nullptr;   //记录最大值的位置，待以后看看能不能改成用cuda

	ResampleType _resampleType = re_Findmax;

protected:
	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void updateDelta2() override;
public:
	void activeOutputValue() override;
	void spreadDeltaToPrevLayer() override;
	void backPropagate(double learnSpeed, double lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;

	void setSubType(ResampleType re) override { _resampleType = re; }
};

