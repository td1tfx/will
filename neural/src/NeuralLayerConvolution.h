#pragma once
#include "NeuralLayer.h"

class NeuralLayerConvolution :
	public NeuralLayer
{
public:
	NeuralLayerConvolution();
	virtual ~NeuralLayerConvolution();

	int kernelCount = 0;
	int kernelRow, kernelCol;

	Matrix* kernelData = nullptr;
	Matrix** kernels = nullptr;
	ConvolutionType _convolutionType = cv_1toN;
	//��Ҫһ�����ӷ�ʽ���󣬿��������鷳
	//Ӧ���ǴӾ���˺ͼ��㷽ʽ���һ�������������Ӧ���ǱȽ�ϡ���
	//�ṩ�������ӷ�ʽ������ˣ��ݴ˼����һ�������
protected:
	void initData2(NeuralLayerInitInfo* info) override {}
	void resetGroupCount2() override {}
	void connetPrevlayer2() override {}
	void updateDelta2() override {}
public:
	void activeOutput() override;
	void spreadDeltaToPrevLayer() override {}
	void updateWeightBias(real learnSpeed, real lambda) override {}
	int saveInfo(FILE* fout) override;
	int loadInfo(real* v, int n) override;

	void setSubType(ConvolutionType cv) override { _convolutionType = cv; }

};

