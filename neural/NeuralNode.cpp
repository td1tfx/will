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
	for (auto &b : prevBonds)
	{
		totalInputValue += b.second.startNode->outputValue * b.second.weight;
		//printf("\t%lf, %lf\n", b.second.startNode->outputValue, b.second.weight);
	}
	//printf("%lf\n",totalInputValue);
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
	if (w == 0)
	{
		w = 1.0*rand() / RAND_MAX;
	}
	//这里内部维护两组连接，其中前连接为主，后连接主要用于计算delta
	//前连接
	auto &pb = this->prevBonds[node];
	pb.startNode = node;
	pb.endNode = this;
	pb.weight = w;
	//后连接
	auto &nb = node->nextBonds[this];
	nb.startNode = this;
	nb.endNode = node;
	nb.weight = w;
}

void NeuralNode::setWeight(NeuralNode* node, double w /*= 0*/)
{
	//if (bonds.find(node) == nullptr)
	{
		prevBonds[node].weight = 0;
	}
}

void NeuralNode::updateWeight(NeuralNode* startNode, NeuralNode* endNode, double learnSpeed, double delta)
{
	double& w = endNode->prevBonds[startNode].weight;
	w += learnSpeed*delta*endNode->outputValue;
	startNode->nextBonds[endNode].weight = w;
}

void NeuralNode::updateDelta(double expect /*= 0*/)
{
	delta = 0;
	if (this->type == Output)
	{
		delta = (expect - outputValue)*feedbackFunction(totalInputValue);
	}
	else
	{
		for (auto& b : nextBonds)
		{
			auto& bond = b.second;
			auto& node = bond.endNode;
			delta += node->delta*bond.weight;
		}
		delta = delta*feedbackFunction(totalInputValue);
	}
}

void NeuralBond::updateWeight(double learnSpeed)
{
	auto startNode = this->startNode, endNode = this->endNode;
	double delta = endNode->delta;
	double& w = endNode->prevBonds[startNode].weight;

	w += learnSpeed*delta*endNode->outputValue;
	startNode->nextBonds[endNode].weight = w;
}
