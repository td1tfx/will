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

void NeuralLayer::createNodes(int nodeAmount, int dataGroupAmount /*= 0*/, NeuralNodeType type /*= Hidden*/, bool haveConstNode /*= false*/)
{
	//！！！注意实际上是多出了一个node，最后一个node的输出无论为何都恒定为1且不与前一层连接
	for (int i = 0; i < nodeAmount; i++)
	{
		auto node = new NeuralNode();
		node->type = type;
		node->setDataGroupAmount(dataGroupAmount);
		if (haveConstNode && i == nodeAmount - 1)
		{
			node->type = Const;
		}
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
