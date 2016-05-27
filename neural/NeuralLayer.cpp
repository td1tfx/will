#include "NeuralLayer.h"


int NeuralLayer::groupAmount;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	if (input) { delete input; }
	if (output) { delete output; }
	if (weight) { delete weight; }
	if (delta) { delete delta; }
	if (expect) { delete expect; }
}


void NeuralLayer::initData(int nodeAmount, int groupAmount)
{
	this->nodeAmount = nodeAmount;
	this->groupAmount = groupAmount;
	int n = nodeAmount*groupAmount;
	if (type != Input)
	{
		input = new d_matrix(nodeAmount,groupAmount);
	}
	output = new d_matrix(nodeAmount, groupAmount);
	delta = new d_matrix(nodeAmount, groupAmount);
	if (mode == HaveConstNode)
	{
		for (int i = 0; i < groupAmount; i++)
		{
			output->getData(nodeAmount - 1, i) = -1;
		}
		//matrixOutput(data, 3, 3);
	}
}

void NeuralLayer::initExpect()
{
	expect = new d_matrix(nodeAmount, groupAmount);
}

//创建weight矩阵
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = startLayer->nodeAmount*endLayer->nodeAmount;
	endLayer->weight = new d_matrix(endLayer->nodeAmount, startLayer->nodeAmount);
	for (int i = 0; i < n; i++)
	{
		endLayer->weight->getData(i) = 1.0 * rand() / RAND_MAX - 0.5;
		//endLayer->weight->getData(i) = 1.0 * i+1;
	}
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
			sum += getOutput(i_node, i_group);
		}
		if (sum == 0) continue;
		for (int i_node = 0; i_node < nodeAmount; i_node++)
		{
			getOutput(i_node, i_group) /= sum;
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
	//this->weight->print();
	//prevLayer->output->print();
	d_matrix::product(this->weight, prevLayer->output, this->input);
	//this->input->print();
	//int m = this->nodeAmount, k = prevLayer->nodeAmount, n = groupAmount;
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, weight, k, prevLayer->output, k, 0, this->input, n);
	for (int i = 0; i < nodeAmount*groupAmount; i++)
	{
		output->getData(i) = activeFunction(input->getData(i));
	}
}

void NeuralLayer::updateDelta()
{
	if (this->type == Output)
	{
		d_matrix::minus(expect, output, delta);
		//deltas[i] *= dactiveFunction(inputValues[i]);
		//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉
	}
	else
	{
		//nextLayer->weight->print();
		//nextLayer->delta->print();
		d_matrix::product(nextLayer->weight, nextLayer->delta, delta, 1, 0, CblasTrans, CblasNoTrans);
		//this->delta->print();
		for (int i = 0; i < nodeAmount*groupAmount; i++)
			delta->getData(i) *= dactiveFunction(input->getData(i));
	}
}

void NeuralLayer::backPropagate(double learnSpeed /*= 0.5*/)
{
	updateDelta();
	//delta->print();
	//prevLayer->output->print();
	d_matrix::product(this->delta, prevLayer->output, this->weight, learnSpeed / groupAmount, 1, CblasNoTrans, CblasTrans);
	//weight->print();
}
