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

//创建神经元
void NeuralLayer::createNodes(int nodeAmount, NeuralNodeType type /*= Hidden*/, bool haveConstNode /*= false*/, int dataAmount /*= 0*/)
{
	//！！！注意实际上是多出了一个node，最后一个node的输出无论为何都恒定为1且不与前一层连接
	nodes.resize(nodeAmount);
	for (int i = 0; i < nodeAmount; i++)
	{
		auto node = new NeuralNode();
		node->type = type;
		node->id = i;
		node->setDataAmount(dataAmount);
		if (haveConstNode && i == nodeAmount - 1)
		{
			node->type = Const;
		}
		nodes[i] = node;
	}
}

//两个神经层所有节点都连接
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	for (auto& startNode : startLayer->nodes)
	{
		for (auto& endNode : endLayer->nodes)
		{
			endNode->connectStart(startNode);
		}
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
