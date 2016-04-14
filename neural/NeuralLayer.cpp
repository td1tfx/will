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
	//������ע��ʵ�����Ƕ����һ��node�����һ��node���������Ϊ�ζ��㶨Ϊ1�Ҳ���ǰһ������
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

//�����񾭲����нڵ㶼����
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
