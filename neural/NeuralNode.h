#pragma once
#include <vector>
#include <functional>

#include "ActiveFunctions.h"

class NeuralBond
{
public:
	double weight;
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
	double totalInputValue;
	void collectInputValue();
	void activeOutputValue();

	std::function<double(double)> activeFunction = ActiveFunctions::sigmoid;
	std::function<double(double)> feedbackFunction = ActiveFunctions::_sigmoid;

};

