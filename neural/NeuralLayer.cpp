#include "NeuralLayer.h"


int NeuralLayer::groupAmount;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	deleteData();
	if (weight) { delete weight; }
}


void NeuralLayer::deleteData()
{
	if (input) { delete input; }
	if (output) { delete output; }
	if (delta) { delete delta; }
	if (expect) { delete expect; }
}

void NeuralLayer::initData(int nodeAmount, int groupAmount, NeuralLayerMode mode)
{
	deleteData();
	this->mode = mode;
	this->inputNodeAmount = nodeAmount;
	this->outputNodeAmount = nodeAmount;
	this->groupAmount = groupAmount;
	if (mode == HaveConstNode)
		this->outputNodeAmount++;
	if (type != Input)
	{
		input = new d_matrix(this->inputNodeAmount,groupAmount);
	}
	output = new d_matrix(this->outputNodeAmount, groupAmount);
	delta = new d_matrix(this->outputNodeAmount, groupAmount);
	if (mode == HaveConstNode)
	{
		for (int i = 0; i < groupAmount; i++)
		{
			output->getData(this->outputNodeAmount - 1, i) = -1;
		}
	}
	//output->print();
}

void NeuralLayer::initExpect()
{
	expect = new d_matrix(outputNodeAmount, groupAmount);
}

//创建weight矩阵
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = endLayer->inputNodeAmount*startLayer->outputNodeAmount;
	endLayer->weight = new d_matrix(endLayer->inputNodeAmount, startLayer->outputNodeAmount);
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
		for (int i_node = 0; i_node < inputNodeAmount; i_node++)
		{
			sum += getOutput(i_node, i_group);
		}
		if (sum == 0) continue;
		for (int i_node = 0; i_node < inputNodeAmount; i_node++)
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
	for (int i = 0; i < inputNodeAmount; i++)
	{
		for (int j = 0; j < groupAmount; j++)
		{
			output->getData(i, j) = activeFunction(input->getData(i, j));
		}
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
		for (int i = 0; i < inputNodeAmount; i++)
		{
			for (int j = 0; j < groupAmount; j++)
			{
				delta->getData(i,j) *= dactiveFunction(input->getData(i,j));
			}
		}
	}
}

void NeuralLayer::backPropagate(double learnSpeed /*= 0.5*/, double lambda /*= 0.1*/)
{
	updateDelta();
	//delta->print();
	//prevLayer->output->print();

	d_matrix::product(delta, prevLayer->output, weight,	learnSpeed / groupAmount, 1, CblasNoTrans, CblasTrans);
	return;
	//这个是加了正则化， 结果不正常
	if (mode == HaveNotConstNode)
	{
		d_matrix::product(delta, prevLayer->output, weight,
			learnSpeed / groupAmount, 1 - lambda*learnSpeed / groupAmount, CblasNoTrans, CblasTrans);
	}
	else
	{
		int m = weight->getRow();
		int n = weight->getCol();
		int k = delta->getCol();
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n-1, k, 
			learnSpeed / groupAmount, delta->getDataPointer(), m, prevLayer->output->getDataPointer(), n,
			1 - lambda*learnSpeed / groupAmount, weight->getDataPointer(), m);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, 1, k,
			learnSpeed / groupAmount, delta->getDataPointer(), m, prevLayer->output->getDataPointer(n - 1, 0), n,
			1, weight->getDataPointer(0, n - 1), m);
	}
	//weight->print();
}
