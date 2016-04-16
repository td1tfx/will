#pragma once
#include <vector>
#include "NeuralNode.h"

class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();


	std::vector<NeuralNode*> nodes;
	NeuralNode*& getNode(int number) { return (nodes.at(number)); }
	int getNodeAmount() { return nodes.size(); };

	void createNodes(int nodeAmount, int dataAmount = 0, NeuralNodeType type = Hidden, bool haveConstNode = false);
	static void connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
};

