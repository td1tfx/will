#pragma once
#include "NeuralLayer.h"
class NeuralLayerPooling :
	public NeuralLayer
{
public:

	//�ڴ���ͼ��ģʽ��ʱ����һ��output�����������ﱻתΪ����

	NeuralLayerPooling();
	virtual ~NeuralLayerPooling();

	int* recordPos = nullptr;   //��¼���ֵ��λ��

	ResampleType _resampleType = re_Max;
	real Weight, Bias;
	//����ֵΪ1
	Matrix* _asBiasMatrix = nullptr;

	int window_w, window_h;  //pooling���ڳߴ�
	int w_stride, h_stride;  //pooling����

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

