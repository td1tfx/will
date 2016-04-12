#include "NeuralNet.h"



NeuralNet::NeuralNet()
{
}


NeuralNet::~NeuralNet()
{
	printf("hihihihi");
	for (auto& layer : this->layers)
	{
		delete layer;
	}
}

void NeuralNet::createLayers(int layerNumber)
{

}
