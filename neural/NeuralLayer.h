#pragma once
#include <vector>
#include "NeuralNode.h"

class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	std::vector<NeuralNode*> nodes;
	void createNodes(int nodeAmount, NeuralNodeType type = hidden);
	void connetTwoLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
};

