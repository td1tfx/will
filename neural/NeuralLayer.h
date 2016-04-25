#pragma once
#include <vector>
#include "NeuralNode.h"

typedef enum 
{
	HaveNotConstNode = 0,
	HaveConstNode = 1,

}NeuralLayerMode;

//�񾭲�
//ע���񾭲�ʵ���ϲ��Ǳ����
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int id;

	std::vector<NeuralNode*> nodes;  //������Ԫ
	std::vector<NeuralNode*>& getNodeVector() { return nodes; }

	NeuralNode*& getNode(int number) { return nodes[number]; }
	int getNodeAmount() { return nodes.size(); };

	void createNodes(int nodeAmount, NeuralNodeType type = Hidden, NeuralLayerMode layerMode = HaveNotConstNode, int dataAmount = 0);
	static void connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
	void markMax();
	void normalized();
};

