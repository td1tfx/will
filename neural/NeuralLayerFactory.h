#pragma once
#include "NeuralLayer.h"
#include "NeuralLayerFull.h"
#include "NeuralLayerConv.h"
#include "NeuralLayerResample.h"

class NeuralLayerFactory
{
public:
	NeuralLayerFactory();
	virtual ~NeuralLayerFactory();

public:
	static NeuralLayer* createLayer(NeuralLayerConnectionMode mode);
	static void destroyLayer(NeuralLayer* layer) { delete layer; };
};

