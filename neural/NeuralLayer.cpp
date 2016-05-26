#include "NeuralLayer.h"

using namespace MatrixFunctions;

int NeuralLayer::groupAmount;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	if (input) { delete[] input; }
	if (output) { delete[] output; }
	if (weight) { delete[] weight; }
	if (delta) { delete[] delta; }
	if (expect) { delete[] expect; }
}


void NeuralLayer::initData(int nodeAmount, int groupAmount)
{
	this->nodeAmount = nodeAmount;
	this->groupAmount = groupAmount;
	int n = nodeAmount*groupAmount;
	if (type != Input)
	{
		input = new double[n];
	}
	output = new double[n];
	delta = new double[n];
	if (mode == HaveConstNode)
	{
		for (int i = 0; i < groupAmount; i++)
		{
			getOutput(nodeAmount - 1, i) = -1;
		}
		//matrixOutput(data, 3, 3);
	}
}

void NeuralLayer::initExpect()
{
	expect = new double[nodeAmount*groupAmount];
}

//创建weight矩阵
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = startLayer->nodeAmount*endLayer->nodeAmount;
	endLayer->weight = new double[n];
	for (int i = 0; i < n; i++)
	{
		endLayer->weight[i] = 1.0 * rand() / RAND_MAX - 0.5;
		//endLayer->weight[i] = 1.0 * i+1;
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
	//matrixOutput(prevLayer->output, groupAmount, prevLayer->nodeAmount);
	d_matrixProduct(weight, prevLayer->output, this->input, this->nodeAmount, prevLayer->nodeAmount, groupAmount, 1, 0, CblasNoTrans, CblasTrans);
	//int m = this->nodeAmount, k = prevLayer->nodeAmount, n = groupAmount;
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, weight, k, prevLayer->output, k, 0, this->input, n);
	for (int i = 0; i < nodeAmount*groupAmount; i++)
	{
		output[i] = activeFunction(input[i]);
	}
}

void NeuralLayer::updateDelta()
{
	if (this->type == Output)
	{
		d_matrixMinus(expect, output, delta, nodeAmount, groupAmount);
		//matrixOutput(expect, groupAmount, nodeAmount);
		//deltas[i] *= dactiveFunction(inputValues[i]);
		//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉
	}
	else
	{
		//这里好像是不对
		d_matrixProduct(nextLayer->weight, nextLayer->delta, this->delta, this->nodeAmount, nextLayer->nodeAmount, groupAmount,
			1, 0, CblasNoTrans, CblasTrans);
		for (int i = 0; i < nodeAmount*groupAmount; i++)
			delta[i] *= dactiveFunction(input[i]);
	}
}

void NeuralLayer::backPropagate(double learnSpeed /*= 0.5*/)
{
	updateDelta();
	//第二个矩阵应该是要转置
	d_matrixProduct(this->delta, prevLayer->output, this->weight,  this->nodeAmount, groupAmount, prevLayer->nodeAmount,
		learnSpeed / groupAmount, 1, CblasNoTrans, CblasTrans);
	//matrixOutput(weight, nodeAmount, prevLayer->nodeAmount);
	//matrixOutput(delta, groupAmount, nodeAmount);
}
