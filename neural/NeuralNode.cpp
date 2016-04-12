#include "NeuralNode.h"



NeuralNode::NeuralNode()
{
}


NeuralNode::~NeuralNode()
{
// 	for (auto &b : bonds)
// 	{
// 	}
}

void NeuralNode::collectInputValue()
{
	double totalInputValue = 0;
	for (auto &b : bonds)
	{
		totalInputValue += b.second.startNode->outputValue * b.second.weight;
	}
}

void NeuralNode::activeOutputValue()
{
	outputValue = activeFunction(totalInputValue);
}

void NeuralNode::setFunctions(std::function<double(double)> _active, std::function<double(double)> _feedback)
{
	activeFunction = _active;
	feedbackFunction = _feedback;
}

void NeuralNode::connect(NeuralNode* node)
{
	auto &b = bonds[node];
	b.startNode = node;
}

