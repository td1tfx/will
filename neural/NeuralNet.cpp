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

void NeuralNet::calOutput(double* input, double* output)
{
	for (int i = 0; i < inputNodeAmount; i++)
	{
		getLayer(0)->getNode(i)->totalInputValue = input[i];
	}
	for (int l = 1; l < layers.size(); l++)
	{
		auto layer = layers[l];
		for (int n = 0; n < layer->nodes.size(); n++)
		{
			auto node = layer->getNode(n);
			node->collectInputValue();
			node->activeOutputValue();
		}
	}
	auto layer = layers.back();
	for (int i = 0; i < outputNodeAmount; i++)
	{
		output[i] = layer->getNode(i)->outputValue;
	}
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

void NeuralNet::setLayers()
{
	auto layer0 = layers.at(0);
	layer0->createNodes(inputNodeAmount, NeuralNodeType::Input);

	auto layer1 = layers.back();
	layer1->createNodes(outputNodeAmount, NeuralNodeType::Output);

	auto layer = layers.at(1);
	layer->createNodes(10);

	layer1->connetPrevlayer(getLayer(1));
	getLayer(1)->connetPrevlayer(layer0);
	//printf("%d,%d,%d\n", layer->getNodeAmount(), layer->getNode(0)->bonds.size(), getLayer(1));
}