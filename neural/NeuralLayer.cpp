#include "NeuralLayer.h"

using namespace MatrixFunctions;

int NeuralLayer::groupAmount;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	if (data) { delete[] data; }
	if (weight) { delete[] weight; }
	if (delta) { delete[] delta; }
	if (expect) { delete[] expect; }
}


void NeuralLayer::initData(int nodeAmount, int groupAmount)
{
	this->nodeAmount = nodeAmount;
	this->groupAmount = groupAmount;
	data = new double[nodeAmount*groupAmount];
	delta = new double[nodeAmount*groupAmount];
	if (HaveConstNode)
	{
		for (int i = 0; i < groupAmount; i++)
		{
			getData(nodeAmount - 1, i) = -1;
		}
		//matrixOutput(data, 3, 3);
	}
}

void NeuralLayer::initExpect()
{
	expect = new double[nodeAmount*groupAmount];
}

//´´½¨weight¾ØÕó
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = startLayer->nodeAmount*endLayer->nodeAmount;
	endLayer->weight = new double[n];
	for (int i = 0; i < n; i++)
	{
		endLayer->weight[i] = 1.0 * rand() / RAND_MAX - 0.5;
		endLayer->weight[i] = 1.0 * i+1;
	}
	matrixOutput(endLayer->weight,  endLayer->nodeAmount, startLayer->nodeAmount);
	endLayer->prevLayer = startLayer;
	startLayer->nextLayer = endLayer;
}

void NeuralLayer::connetPrevlayer(NeuralLayer* prevLayer)
{
	connetLayer(prevLayer, this);
}

void NeuralLayer::connetNextlayer(NeuralLayer* nextLayer)
{
	connetLayer(this, nextLayer);
}

void NeuralLayer::markMax(int groupid)
{
// 	if (getNodeAmount() <= 0) return;
// 	for (int i_data = 0; i_data < getNode(0)->getDataAmount(); i_data++)
// 	{
// 		double now_max = getNode(0)->getOutput(i_data);
// 		getNode(0)->setOutput(0, i_data);
// 		int pos = 0;
// 		for (int i_node = 1; i_node < getNodeAmount(); i_node++)
// 		{
// 			if (now_max <= getNode(i_node)->getOutput(i_data))
// 			{
// 				now_max = getNode(i_node)->getOutput(i_data);
// 				pos = i_node;
// 			}
// 			getNode(i_node)->setOutput(0, i_data);
// 		}
// 		getNode(pos)->setOutput(1, i_data);
// 	}
}

void NeuralLayer::normalized()
{
	for (int i_group = 0; i_group < this->groupAmount; i_group++)
	{
		double sum = 0;
		for (int i_node = 0; i_node < nodeAmount; i_node++)
		{
			sum += getData(i_node, i_group);
		}
		if (sum == 0) continue;
		for (int i_node = 0; i_node < nodeAmount; i_node++)
		{
			getData(i_node, i_group) /= sum;
		}
	}
}

void NeuralLayer::setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive)
{
	activeFunction = _active;
	dactiveFunction = _dactive;
}

void NeuralLayer::activeOutputValue()
{
	//matrixOutput(prevLayer->data, prevLayer->nodeAmount, groupAmount);
	d_matrixProduct(weight, prevLayer->data, this->data, this->nodeAmount, prevLayer->nodeAmount, groupAmount);

	for (int i = 0; i < nodeAmount*groupAmount; i++)
	{
		data[i] = activeFunction(data[i]);
	}
}

void NeuralLayer::updateDelta()
{

}

void NeuralLayer::backPropagate(double learnSpeed /*= 0.5*/)
{

}
