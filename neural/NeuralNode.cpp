#include "NeuralNode.h"



NeuralNode::NeuralNode()
{
}


NeuralNode::~NeuralNode()
{
}

void NeuralNode::collectInputValue()
{
	double totalInputValue = 0;
	for (auto &b : bonds)
	{
		totalInputValue += b.startNode->outputValue * b.weight;
	}
}

void NeuralNode::activeOutputValue()
{
	outputValue = activeFunction(totalInputValue);
}
