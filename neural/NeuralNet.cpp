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

void NeuralNet::learn(double* input, double* output)
{
	auto o = new double[outputNodeAmount];
	activeOutputValue(input, o);
	printf("output = %lf, %lf\n", o[0], output[0]);

	//这里是输出层
	auto layer_output = layers.back();
	for (int j = 0; j < layer_output->nodes.size(); j++)
	{
		auto node = layer_output->getNode(j);
		node->updateDelta(output[j]);
		for (auto b : node->prevBonds)
		{
			auto& bond = b.second;
			bond.updateWeight(learnSpeed);
		}
	}
	for (int l = layers.size() - 1; l > 0; l--)
	{
		auto layer = layers[l];
		for (int j = 0; j < layer->nodes.size(); j++)
		{
			auto node = layer->getNode(j);
			node->updateDelta();
			for (auto b : node->prevBonds)
			{
				auto& bond = b.second;
				bond.updateWeight(learnSpeed);
			}
		}
	}
	delete o;

}

void NeuralNet::train()
{
	//测试中，暂时只算一个
	for (int i = 0; i < 1; i++)
	{
		for (int j = 1; j <= 10;j++)
		learn(inputData + i*inputNodeAmount, outputData + i*outputNodeAmount);
	}
}

//注意：这里按照前面的设计应该是逐步回溯计算，使用栈保存计算的顺序
void NeuralNet::activeOutputValue(double* input, double* output)
{
	for (int i = 0; i < inputNodeAmount; i++)
	{
		getLayer(0)->getNode(i)->outputValue = input[i];
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

//这里的处理可能不是很好
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
	int k1 = 0, k2 = 0;
	inputData = new double[inputNodeAmount * dataGroupAmount];
	outputData = new double[outputNodeAmount * dataGroupAmount];
	for (int i = 1; i <= dataGroupAmount; i++)
	{
		for (int j = 1; j <= inputNodeAmount; j++)
		{
			inputData[k1++] = v.at(k++);
		}
		for (int j = 1; j <= outputNodeAmount; j++)
		{
			outputData[k2++] = v.at(k++);
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
	for (auto node : layer->nodes)
	{
		node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
	}
	for (auto node : layer1->nodes)
	{
		node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
	}

	layer1->connetPrevlayer(getLayer(1));
	getLayer(1)->connetPrevlayer(layer0);
	//printf("%d,%d,%d\n", layer->getNodeAmount(), layer->getNode(0)->bonds.size(), getLayer(1));
}