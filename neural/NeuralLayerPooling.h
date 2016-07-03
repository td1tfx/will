#pragma once
#include "NeuralLayer.h"
class NeuralLayerPooling :
	public NeuralLayer
{
public:

	//�ڴ���ͼ��ģʽ��ʱ����һ��output�����������ﱻתΪ����

	NeuralLayerPooling();
	virtual ~NeuralLayerPooling();

	int* maxPos = nullptr;   //��¼���ֵ��λ��

	ResampleType _resampleType = re_Max;
	double Weight, Bias;
	//����ֵΪ1
	Matrix* _asBiasMatrix = nullptr;

	int window_w, window_h;  //pooling���ڳߴ�
	int w_stride, h_stride;  //pooling����

protected:
	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void backPropagateDelta2() override;
public:
	void activeForwardOutput() override;
	void spreadDeltaToPrevLayer() override;
	void updateWeightBias(double learnSpeed, double lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;

	void setSubType(ResampleType re) override { _resampleType = re; }
};

