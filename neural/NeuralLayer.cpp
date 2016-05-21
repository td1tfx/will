#include "NeuralLayer.h"

using namespace MatrixFunctions;

int NeuralLayer::groupAmount;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	if (data)
		delete []data;
	if (weight)
		delete []weight;
}


void NeuralLayer::initData(int nodeAmount, int groupAmount)
{
	this->nodeAmount = nodeAmount;
	this->groupAmount = groupAmount;
	data = new double[nodeAmount*groupAmount];
}

//´´½¨weight¾ØÕó
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = startLayer->nodeAmount*endLayer->nodeAmount;
	endLayer->weight = new double[n];
	for (int i = 0; i < n; i++)
	{
		endLayer->weight[i] = 1.0 * rand() / RAND_MAX - 0.5;
	}
}

void NeuralLayer::connetPrevlayer(NeuralLayer* prevLayer)
{
	connetLayer(prevLayer, this);
}

void NeuralLayer::connetNextlayer(NeuralLayer* nextLayer)
{
	connetLayer(this, nextLayer);
}

void NeuralLayer::markMax(int groupid)
{
// 	if (getNodeAmount() <= 0) return;
// 	for (int i_data = 0; i_data < getNode(0)->getDataAmount(); i_data++)
// 	{
// 		double now_max = getNode(0)->getOutput(i_data);
// 		getNode(0)->setOutput(0, i_data);
// 		int pos = 0;
// 		for (int i_node = 1; i_node < getNodeAmount(); i_node++)
// 		{
// 			if (now_max <= getNode(i_node)->getOutput(i_data))
// 			{
// 				now_max = getNode(i_node)->getOutput(i_data);
// 				pos = i_node;
// 			}
// 			getNode(i_node)->setOutput(0, i_data);
// 		}
// 		getNode(pos)->setOutput(1, i_data);
// 	}
}

void NeuralLayer::normalized()
{
	for (int i_group = 0; i_group < this->groupAmount; i_group++)
	{
		double sum = 0;
		for (int i_node = 0; i_node < nodeAmount; i_node++)
		{
			sum += getValue(i_node, i_group);
		}
		if (sum == 0) continue;
		for (int i_node = 0; i_node < nodeAmount; i_node++)
		{
			setValue(getValue(i_node, i_group) / sum, i_node, i_group);
		}
	}
}

void NeuralLayer::activeOutputValue()
{
	d_matrixProduct(weight, prevLayer->data, this->data, prevLayer->nodeAmount, this->nodeAmount, groupAmount);

	for (int i = 0; i < nodeAmount*groupAmount; i++)
	{
		data[i] = activeFunction(data[i]);
	}
}
