#pragma once
#include <vector>
class NeuralBond
{
	double value;
	class NeuralNode* startNode;
	class NeuralNode* endNode;
};

class NeuralNode
{

public:
	NeuralNode();
	virtual ~NeuralNode();

	std::vector<NeuralBond> bonds;
	double outputValue;

};

