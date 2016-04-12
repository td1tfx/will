#include "NeuralNet.h"



NeuralNet::NeuralNet()
{
}


NeuralNet::~NeuralNet()
{
	for (auto& layer : this->layers)
	{
		delete layer;
	}
	if (inputData)
	{
		delete inputData;
	}
	if (outputData)
	{
		delete outputData;
	}
}

void NeuralNet::createLayers(int layerNumber)
{
	for (int i = 1; i <= layerNumber; i++)
	{
		auto layer = new NeuralLayer();
		layers.push_back(layer);
	}
}

void NeuralNet::learn(void* input, void* output)
{

}

void NeuralNet::train(void* input, void* output)
{

}

void NeuralNet::setLayers()
{
	auto layer0 = layers.at(0);
	layer0->createNodes(inputNodeAmount, NeuralNodeType::Input);
	
	auto layer1 = layers.back();
	layer1->createNodes(outputNodeAmount, NeuralNodeType::Output);
	
	layer1->connetPrevlayer(getLayer(1));
	getLayer(1)->connetPrevlayer(layer0);
}

void NeuralNet::readData(std::string& filename)
{
	std::string str = readStringFromFile(filename);
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	inputNodeAmount = int(v.at(0));
	outputNodeAmount = int(v.at(1));

	dataGroupAmount = (n - 2) / (inputNodeAmount + outputNodeAmount);

	int k = 2;
	inputData = new double[inputNodeAmount * dataGroupAmount];
	outputData = new double[outputNodeAmount * dataGroupAmount];
	for (int i = 1; i <= dataGroupAmount; i++)
	{
		int k1 = 0, k2 = 0;
		for (int j = 1; j <= inputNodeAmount; j++)
		{
			((double*)inputData)[k1++] = v.at(k++);
		}
		for (int j = 1; j <= outputNodeAmount; j++)
		{
			((double*)outputData)[k2++] = v.at(k++);
		}
	}
}

