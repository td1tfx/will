#pragma once
#include "NeuralLayer.h"
class NeuralLayerConv :
	public NeuralLayer
{
public:
	NeuralLayerConv();
	virtual ~NeuralLayerConv();

	virtual void initData2(int x1, int x2) {}
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2() {}
	virtual void updateDelta2() {}

	virtual void activeOutputValue() {}
	virtual void spreadDeltaToPrevLayer() {}
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(double* v, int n) { return 0; }

};

