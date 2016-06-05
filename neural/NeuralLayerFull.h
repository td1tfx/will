#pragma once
#include "NeuralLayer.h"
class NeuralLayerFull :
	public NeuralLayer
{
public:
	int hitls;

	NeuralLayerFull();
	virtual ~NeuralLayerFull();

	virtual void initData(int nodeCount, int groupCount, NeuralLayerType type = Hidden);
	virtual void resetData(int groupCount);
	virtual void connetPrevlayer(NeuralLayer* prevLayer);
	virtual void activeOutputValue();
	virtual void updateDelta();
	virtual void backPropagate(double learnSpeed, double lambda);

	virtual int saveInfo(FILE* fout);
	virtual int readInfo(double* v, int n);
};

