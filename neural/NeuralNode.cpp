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
	totalInputValue = 0;
	for (auto &b : bonds)
	{
		totalInputValue += b.second.startNode->outputValue * b.second.weight;
		printf("\t%lf, %lf\n", b.second.startNode->outputValue, b.second.weight);
	}
	printf("%lf\n",totalInputValue);
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

void NeuralNode::connect(NeuralNode* node, double w /*= 0*/)
{
	auto &b = bonds[node];
	b.startNode = node;
	b.endNode = this;
	b.weight = w;
	if (w == 0)
		b.weight = 1.0*rand() / RAND_MAX;
}

void NeuralNode::setWeight(NeuralNode* node, double w /*= 0*/)
{
	//if (bonds.find(node) == nullptr)
	{
		bonds[node].weight = 0;
	}
}

