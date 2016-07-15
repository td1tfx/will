#pragma once
#include "Layer.h"

class LayerPooling :
	public Layer
{
public:
	LayerPooling();
	virtual ~LayerPooling();

	int* recordPos = nullptr;   //��¼���ֵ��λ��

	PoolingType _resampleType = pl_Max;
	real Weight, Bias;
	//����ֵΪ1
	Matrix* _asBiasMatrix = nullptr;

	int window_w, window_h;  //pooling���ڳߴ�
	int stride_w, stride_h;  //pooling����

protected:
	void initData2(LayerInitInfo* info) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void activeBackward2() override;
public:
	void activeForward() override;
	void spreadDeltaToPrevLayer() override;
	void updateParameters(real learnSpeed, real lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(real* v, int n) override;

	void setSubType(PoolingType re) override { _resampleType = re; }
};

