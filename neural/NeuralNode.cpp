#include "NeuralNode.h"
#include <cstdlib>

NeuralNode::NeuralNode()
{
	dataAmount = 0;
}


NeuralNode::~NeuralNode()
{
 	for (auto &b : prevBonds)
 	{
		delete b.second;
 	}
}

void NeuralNode::setExpect(double x, int i /*= -1*/)
{
	this->expect = x;
	if (i >= 0 && i < dataAmount)
	{
		this->expects[i] = x;
	}
}

void NeuralNode::setInput(double x, int i /*= -1*/)
{
	this->inputValue = x;
	if (i >= 0 && i < dataAmount)
	{
		this->inputValues[i] = x;
	}
}

void NeuralNode::setOutput(double x, int i /*= -1*/)
{
	if (type == Const) return;
	this->outputValue = x;
	if (i >= 0 && i < dataAmount)
	{
		this->outputValues[i] = x;
	}
}

double NeuralNode::getOutput(int i /*= -1*/)
{
	if (i < 0) return outputValue;
	else return outputValues[i];
}

//���ｫ�����ݵ����д����һ�𣬿�����Ҫ����
void NeuralNode::collectInputValue()
{
	inputValue = 0;
	for (int i = 0; i < dataAmount; i++)
	{
		inputValues[i] = 0;
	}
	if (type == Const)
		return;
	for (auto &b : prevBonds)
	{
		inputValue += b.second->startNode->outputValue * b.second->weight;
		for (int i = 0; i < dataAmount; i++)
		{
			inputValues[i] += b.second->startNode->outputValues[i] * b.second->weight;
		}
		//printf("\t%lf, %lf\n", b.second.startNode->outputValue, b.second.weight);
	}
	//printf("%lf\n",totalInputValue);
}

//ͬ��
void NeuralNode::activeOutputValue()
{
	if (type == Const)
	{
		for (int i = 0; i < dataAmount; i++)
		{
			outputValues[i] = -1;
		}
		return;
	}
	outputValue = activeFunction(inputValue);
	for (int i = 0; i < dataAmount; i++)
	{
		outputValues[i] = activeFunction(inputValues[i]);
	}
}

void NeuralNode::active()
{
	collectInputValue();
	activeOutputValue();
}

void NeuralNode::setFunctions(std::function<double(double)> _active, std::function<double(double)> _feedback)
{
	activeFunction = _active;
	dactiveFunction = _feedback;
}

void NeuralNode::connect(NeuralNode* start, NeuralNode* end, double w /*= 0*/)
{
	if (w == 0)
	{
		w = 1.0 * rand() / RAND_MAX - 0.5;
	}
	auto bond = new NeuralBond();
	bond->startNode = start;
	bond->endNode = end;
	bond->weight = w;
	//�����ڲ�ά���������ӣ�����ǰ����Ϊ������������Ҫ���ڼ���delta
	//ǰ����
	end->prevBonds[start] = bond;
	//������
	start->nextBonds[end] = bond;
}

void NeuralNode::connectStart(NeuralNode* node, double w /*= 0*/)
{
	connect(node, this, w);
}

void NeuralNode::connectEnd(NeuralNode* node, double w /*= 0*/)
{
	connect(this, node, w);
}

void NeuralNode::setWeight(NeuralNode* node, double w /*= 0*/)
{
	//if (bonds.find(node) == nullptr)
	{
		prevBonds[node]->weight = w;
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
	for (int i = 0; i < dataAmount; i++)
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

void NeuralNode::BackPropagation(double learnSpeed /*= 0.5*/)
{
	updateDelta();
	for (auto b : prevBonds)
	{
		auto& bond = b.second;
		bond->updateWeight(learnSpeed);
	}
}

void NeuralNode::setDataGroupAmount(int n)
{
	dataAmount = n;
	inputValues.resize(n);
	outputValues.resize(n);
	expects.resize(n);
	deltas.resize(n);
	setVectorValue(outputValues);

}

void NeuralBond::updateWeight(double learnSpeed)
{
	auto startNode = this->startNode, endNode = this->endNode;
	double& w = endNode->prevBonds[startNode]->weight;
	int n = startNode->dataAmount;
	if (n <= 0)
	{
		auto& delta = endNode->delta;
		w += learnSpeed*delta*startNode->outputValue;
	}
	else
	{
		double delta_w = 0;
		auto& deltas = endNode->deltas;
		for (int i = 0; i < n; i++)
		{
			delta_w += learnSpeed*deltas[i]*startNode->outputValues[i];
		}
		delta_w /= n;
		w += delta_w;
	}
	//startNode->nextBonds[endNode]->weight = w;
}
