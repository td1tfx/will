#include "NeuralLayerFactory.h"



NeuralLayerFactory::NeuralLayerFactory()
{
}


NeuralLayerFactory::~NeuralLayerFactory()
{
}

NeuralLayer* NeuralLayerFactory::createLayer(NeuralLayerConnectionType mode)
{
	NeuralLayer* layer = nullptr;
	switch (mode)
	{
	case lc_Full:
		layer = new NeuralLayerFull();
		break;
	case lc_Convolution:
		layer = new NeuralLayerConvolution();
		break;
	case lc_Resample:
		layer = new NeuralLayerPooling();
		break;
	default:
		break;
	}
	return layer;
}
