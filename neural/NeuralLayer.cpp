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

//������Ԫ
void NeuralLayer::createNodes(int nodeAmount, NeuralNodeType type /*= Hidden*/, NeuralLayerMode layerMode /*= HaveNotConstNode*/, int dataAmount /*= 0*/)
{
	//������ע��ʵ�����Ƕ����һ��node�����һ��node���������Ϊ�ζ��㶨Ϊ1�Ҳ���ǰһ������
	nodes.resize(nodeAmount);
	for (int i = 0; i < nodeAmount; i++)
	{
		auto node = new NeuralNode();
		node->type = type;
		node->id = i;
		node->setDataAmount(dataAmount);
		if (layerMode == HaveConstNode && i == nodeAmount - 1)
		{
			node->type = Const;
		}
		nodes[i] = node;
	}
}

//�����񾭲����нڵ㶼����
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
