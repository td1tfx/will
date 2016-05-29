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
			if (endNode->type != Const)
			{
				endNode->connectStart(startNode);
			}
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

void NeuralLayer::markMax()
{
	if (getNodeAmount() <= 0) return;
	for (int i_data = 0; i_data < getNode(0)->getDataAmount(); i_data++)
	{
		double now_max = getNode(0)->getOutput(i_data);
		getNode(0)->setOutput(0, i_data);
		int pos = 0;
		for (int i_node = 1; i_node < getNodeAmount(); i_node++)
		{
			if (now_max <= getNode(i_node)->getOutput(i_data))
			{
				now_max = getNode(i_node)->getOutput(i_data);
				pos = i_node;
			}
			getNode(i_node)->setOutput(0, i_data);
		}
		getNode(pos)->setOutput(1, i_data);
	}
}

void NeuralLayer::normalized()
{
	for (int i = 0; i < NeuralNode::dataAmount; i++)
	{
		double sum = 0;
		for (auto& node : nodes)
		{
			sum += node->outputValues[i];
		}
		if (sum == 0) continue;
		for (auto& node : nodes)
		{
			node->outputValues[i] /= sum;
		}
	}
}
