#include "NeuralLayer.h"


int NeuralLayer::groupAmount;
int NeuralLayer::step;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	deleteData();
	if (weight) { delete weight; }
	if (bias) { delete bias; }
}


void NeuralLayer::deleteData()
{
	if (input) { delete input; }
	if (output) { delete output; }
	if (delta) { delete delta; }
	if (expect) { delete expect; }
	if (bias_as) { delete bias_as; }
}

void NeuralLayer::initData(int nodeAmount, int groupAmount, NeuralLayerType type /*= Hidden*/)
{
	deleteData();
	this->type = type;
	this->nodeAmount = nodeAmount;
	this->groupAmount = groupAmount;

	if (type == Input)
	{
		output = new d_matrix(nodeAmount, groupAmount, false);
	}
	else
	{
		output = new d_matrix(nodeAmount, groupAmount);
		input = new d_matrix(nodeAmount, groupAmount);
	}
	if (type == Output)
	{
		expect = new d_matrix(nodeAmount, groupAmount, false);
	}
	
	delta = new d_matrix(nodeAmount, groupAmount);
	bias_as = new d_matrix(groupAmount, 1);
	bias_as->initData(1);
	//output->print();
}

void NeuralLayer::resetData(int groupAmount)
{
	initData(this->nodeAmount, groupAmount, this->type);
}

//创建weight矩阵
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = endLayer->nodeAmount*startLayer->nodeAmount;
	endLayer->weight = new d_matrix(endLayer->nodeAmount, startLayer->nodeAmount);
	endLayer->weight->initRandom();
	endLayer->bias = new d_matrix(endLayer->nodeAmount, 1);
	endLayer->bias->initRandom();
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

//每组的最大值标记为1，其余标记为0
void NeuralLayer::markMax()
{
	if (nodeAmount <= 0) return;
	for (int i_data = 0; i_data < nodeAmount; i_data++)
	{
		double now_max = getOutput(0, i_data);
		getOutput(0, i_data) = 0;
		int pos = 0;
		for (int i_node = 1; i_node < nodeAmount; i_node++)
		{
			if (now_max <= getOutput(i_node, i_data))
			{
				now_max = getOutput(i_node, i_data);
				pos = i_node;
			}
			getOutput(i_node, i_data) = 0;
		}
		getOutput(pos, i_data) = 1;
	}
}

//归一化，暂时无用，不考虑
void NeuralLayer::normalized()
{
	for (int i_group = 0; i_group < groupAmount; i_group++)
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
	d_matrix::cpyData(input, bias);
	input->expand();
	//input->print();
	d_matrix::product(this->weight, prevLayer->output, this->input, 1, 1);
	//this->input->print();
	for (int i = 0; i < nodeAmount; i++)
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
		//这里看起来不对，常数节点没参与运算
		for (int i = 0; i < nodeAmount; i++)
		{
			for (int j = 0; j < groupAmount; j++)
			{
				delta->getData(i, j) *= dactiveFunction(input->getData(i, j));
			}
		}
	}
}

void NeuralLayer::backPropagate(double learnSpeed /*= 0.5*/, double lambda /*= 0.1*/)
{
	updateDelta();
	//lambda = 0.0;
	d_matrix::product(delta, prevLayer->output, weight,
		learnSpeed / groupAmount, 1 - lambda * learnSpeed / groupAmount, CblasNoTrans, CblasTrans);
	d_matrix::productVector(delta, bias_as, bias, learnSpeed / groupAmount, 1, CblasNoTrans);

}
