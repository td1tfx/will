#pragma once
#include "NeuralLayer.h"
class NeuralLayerPooling :
	public NeuralLayer
{
public:

	//在处理图像模式的时候，上一层output的向量在这里被转为矩阵

	NeuralLayerPooling();
	virtual ~NeuralLayerPooling();

	int* recordPos = nullptr;   //记录最大值的位置

	ResampleType _resampleType = re_Max;
	real Weight, Bias;
	//所有值为1
	Matrix* _asBiasMatrix = nullptr;

	int window_w, window_h;  //pooling窗口尺寸
	int w_stride, h_stride;  //pooling步长

protected:
	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void updateDelta2() override;
public:
	void activeOutput() override;
	void spreadDeltaToPrevLayer() override;
	void updateWeightBias(real learnSpeed, real lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(real* v, int n) override;

	void setSubType(ResampleType re) override { _resampleType = re; }
};

