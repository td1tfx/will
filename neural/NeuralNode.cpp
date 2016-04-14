#include "NeuralNode.h"



NeuralNode::NeuralNode()
{
	dataGroupAmount = 0;
}


NeuralNode::~NeuralNode()
{
 	for (auto &b : prevBonds)
 	{
		delete b.second;
 	}
}

void NeuralNode::setExpect(double expect, int i /*= -1*/)
{
	this->expect = expect;
	if (i >= 0 && i < dataGroupAmount)
	{
		this->expects[i] = expect;
	}
}

//这里将多数据的情况写在了一起，可能需要调整
void NeuralNode::collectInputValue()
{
	inputValue = 0;
	for (int i = 0; i < dataGroupAmount; i++)
	{
		inputValues[i] = 0;
	}
	for (auto &b : prevBonds)
	{
		inputValue += b.second->startNode->outputValue * b.second->weight;
		for (int i = 0; i < dataGroupAmount; i++)
		{
			inputValues[i] += b.second->startNode->outputValues[i] * b.second->weight;
		}
		//printf("\t%lf, %lf\n", b.second.startNode->outputValue, b.second.weight);
	}
	//printf("%lf\n",totalInputValue);
}

//同上
void NeuralNode::activeOutputValue()
{
	outputValue = activeFunction(inputValue);
	for (int i = 0; i < dataGroupAmount; i++)
	{
		outputValues[i] = activeFunction(inputValues[i]);
	}
}

void NeuralNode::setFunctions(std::function<double(double)> _active, std::function<double(double)> _feedback)
{
	activeFunction = _active;
	dactiveFunction = _feedback;
}

void NeuralNode::connect(NeuralNode* node, double w /*= 0*/)
{
	if (w == 0)
	{
		w = 1.0*rand() / RAND_MAX-0.5;
	}
	auto bond = new NeuralBond();
	bond->startNode = node;
	bond->endNode = this;
	bond->weight = w;
	//这里内部维护两组连接，其中前连接为主，后连接主要用于计算delta
	//前连接
	this->prevBonds[node] = bond;
	//后连接
	node->nextBonds[this] = bond;

}

void NeuralNode::setWeight(NeuralNode* node, double w /*= 0*/)
{
	//if (bonds.find(node) == nullptr)
	{
		prevBonds[node]->weight = 0;
	}
}

void NeuralNode::updateWeight(NeuralNode* startNode, NeuralNode* endNode, double learnSpeed, double delta)
{
	double& w = endNode->prevBonds[startNode]->weight;
	w += learnSpeed*delta*endNode->outputValue;
	//startNode->nextBonds[endNode]->weight = w;
}

void NeuralNode::updateOneDelta()
{
	delta = 0;
	if (this->type == Output)
	{
		delta = (expect - outputValue)*dactiveFunction(inputValue);
	}
	else
	{
		for (auto& b : nextBonds)
		{
			auto& bond = b.second;
			auto& node = bond->endNode;
			delta += node->delta*bond->weight;
		}
		delta = delta*dactiveFunction(inputValue);
	}
}

void NeuralNode::updateDelta()
{
	this->updateOneDelta();
	for (int i = 0; i < dataGroupAmount; i++)
	{
		deltas[i] = 0;
		if (this->type == Output)
		{
			deltas[i] = (expects[i] - outputValues[i])*dactiveFunction(inputValues[i]);
		}
		else
		{
			for (auto& b : nextBonds)
			{
				auto& bond = b.second;
				auto& node = bond->endNode;
				deltas[i] += node->deltas[i] *bond->weight;
			}
			deltas[i] *= dactiveFunction(inputValues[i]);
		}
	}
}

void NeuralNode::setDataGroupAmount(int n)
{
	dataGroupAmount = n;
	inputValues.resize(n);
	outputValues.resize(n);
	expects.resize(n);
	deltas.resize(n);
}

void NeuralBond::updateWeight(double learnSpeed)
{
	auto startNode = this->startNode, endNode = this->endNode;
	double& w = endNode->prevBonds[startNode]->weight;
	int n = startNode->dataGroupAmount;
	if (n <= 0)
	{
		auto& delta = endNode->delta;
		w += learnSpeed*delta*endNode->outputValue;
	}
	else
	{
		double delta_w = 0;
		auto& deltas = endNode->deltas;
		for (int i = 0; i < n; i++)
		{
			delta_w += learnSpeed*deltas[i]*endNode->outputValues[i];
		}
		delta_w /= n;
		w += delta_w;
	}
	startNode->nextBonds[endNode]->weight = w;
}
