#include "NeuralLayer.h"



NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	for (auto& node : nodes)
	{
		delete node;
	}
}

void NeuralLayer::createNodes(int nodeAmount, int dataGroupAmount, NeuralNodeType type /*= Hidden*/)
{
	for (int i = 1; i <= nodeAmount; i++)
	{
		auto node = new NeuralNode();
		node->type = type;
		node->setDataGroupAmount(dataGroupAmount);
		nodes.push_back(node);
	}
}

//两个神经层所有节点都连接
void NeuralLayer::connetTwoLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	for (auto& startNode : startLayer->nodes)
	{
		for (auto& endNode : endLayer->nodes)
		{
			endNode->connect(startNode);
		}
	}
}

void NeuralLayer::connetPrevlayer(NeuralLayer* prevLayer)
{
	connetTwoLayer(prevLayer, this);
}

void NeuralLayer::connetNextlayer(NeuralLayer* nextLayer)
{
	connetTwoLayer(this, nextLayer);
}
