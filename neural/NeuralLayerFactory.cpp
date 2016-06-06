#include "NeuralLayerFactory.h"



NeuralLayerFactory::NeuralLayerFactory()
{
}


NeuralLayerFactory::~NeuralLayerFactory()
{
}

NeuralLayer* NeuralLayerFactory::createLayer(NeuralLayerConnectionMode mode)
{
	NeuralLayer* layer = nullptr;
	switch (mode)
	{
	case FullConnection:
		layer = new NeuralLayerFull();
		break;
	case Convolution:
		layer = new NeuralLayerConv();
		break;
	case Resample:
		layer = new NeuralLayerResample();
		break;
	default:
		break;
	}
	return layer;
}
