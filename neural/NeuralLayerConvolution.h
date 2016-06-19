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

	d_matrix* kernelData = nullptr;
	d_matrix** kernels = nullptr;
	ConvolutionType _convolutionType = cv_1toN;
	//需要一个连接方式矩阵，看起来很麻烦
	//应该是从卷积核和计算方式算出一个矩阵，这个矩阵应该是比较稀疏的
	//提供的是连接方式，卷积核，据此计算出一个大矩阵
protected:
	void initData2(int x1, int x2) override {}
	void resetGroupCount2() override {}
	void connetPrevlayer2() override {}
	void updateDelta2() override {}
public:
	void activeOutputValue() override;
	void spreadDeltaToPrevLayer() override {}
	void backPropagate(double learnSpeed, double lambda) override {}
	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;

	void setSubType(ConvolutionType cv) override { _convolutionType = cv; }

};

