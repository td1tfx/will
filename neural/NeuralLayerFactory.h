#pragma once
#include "NeuralLayer.h"
#include "NeuralLayerFull.h"
#include "NeuralLayerConvolution.h"
#include "NeuralLayerResample.h"

class NeuralLayerFactory
{
public:
	NeuralLayerFactory();
	virtual ~NeuralLayerFactory();
public:
	static NeuralLayer* createLayer(NeuralLayerConnectionType mode);
	static void destroyLayer(NeuralLayer* layer) { delete layer; };

};

