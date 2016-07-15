#pragma once
#include "NeuralLayer.h"

class NeuralLayerPooling :
	public NeuralLayer
{
public:
	NeuralLayerPooling();
	virtual ~NeuralLayerPooling();

	int* recordPos = nullptr;   //记录最大值的位置

	ResampleType _resampleType = re_Max;
	real Weight, Bias;
	//所有值为1
	Matrix* _asBiasMatrix = nullptr;

	int window_w, window_h;  //pooling窗口尺寸
	int stride_w, stride_h;  //pooling步长

protected:
	void initData2(NeuralLayerInitInfo* info) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void activeBackward2() override;
public:
	void activeForward() override;
	void spreadDeltaToPrevLayer() override;
	void updateParameters(real learnSpeed, real lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(real* v, int n) override;

	void setSubType(ResampleType re) override { _resampleType = re; }
};

