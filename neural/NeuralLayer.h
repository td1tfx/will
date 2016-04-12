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

	void createNodes(int nodeAmount, NeuralNodeType type = Hidden);
	static void connetTwoLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
};

