#include "NeuralLayer.h"


int NeuralLayer::GroupCount;
int NeuralLayer::Step;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	deleteData();
	if (WeightMatrix) { delete WeightMatrix; }
	if (BiasVector) { delete BiasVector; }
}


void NeuralLayer::deleteData()
{
	if (InputMatrix) { delete InputMatrix; }
	if (OutputMatrix) { delete OutputMatrix; }
	if (DeltaMatrix) { delete DeltaMatrix; }
	if (ExpectMatrix) { delete ExpectMatrix; }
	if (_asBiasVector) { delete _asBiasVector; }
}

void NeuralLayer::initData(int nodeCount, int groupCount, NeuralLayerType type /*= Hidden*/)
{
	deleteData();
	this->Type = type;
	this->NodeCount = nodeCount;
	this->GroupCount = groupCount;

	if (type == Input)
	{
		OutputMatrix = new d_matrix(nodeCount, groupCount, false);
	}
	else
	{
		OutputMatrix = new d_matrix(nodeCount, groupCount);
		InputMatrix = new d_matrix(nodeCount, groupCount);
	}
	if (type == Output)
	{
		ExpectMatrix = new d_matrix(nodeCount, groupCount, false);
	}
	
	DeltaMatrix = new d_matrix(nodeCount, groupCount);
	_asBiasVector = new d_matrix(groupCount, 1);
	_asBiasVector->initData(1);
	//output->print();
}

void NeuralLayer::resetData(int groupCount)
{
	initData(this->NodeCount, groupCount, this->Type);
}

//创建weight矩阵
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	int n = endLayer->NodeCount*startLayer->NodeCount;
	endLayer->WeightMatrix = new d_matrix(endLayer->NodeCount, startLayer->NodeCount);
	endLayer->WeightMatrix->initRandom();
	endLayer->BiasVector = new d_matrix(endLayer->NodeCount, 1);
	endLayer->BiasVector->initRandom();
	endLayer->PrevLayer = startLayer;
	startLayer->NextLayer = endLayer;
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
	if (NodeCount <= 0) return;
	for (int i_group = 0; i_group < GroupCount; i_group++)
	{
		double now_max = getOutputValue(0, i_group);
		getOutputValue(0, i_group) = 0;
		int pos = 0;
		for (int i_node = 1; i_node < NodeCount; i_node++)
		{
			if (now_max <= getOutputValue(i_node, i_group))
			{
				now_max = getOutputValue(i_node, i_group);
				pos = i_node;
			}
			getOutputValue(i_node, i_group) = 0;
		}
		getOutputValue(pos, i_group) = 1;
	}
}

//归一化，暂时无用，不考虑
void NeuralLayer::normalized()
{
	for (int i_group = 0; i_group < GroupCount; i_group++)
	{
		double sum = 0;
		for (int i_node = 0; i_node < NodeCount; i_node++)
		{
			sum += getOutputValue(i_node, i_group);
		}
		if (sum == 0) continue;
		for (int i_node = 0; i_node < NodeCount; i_node++)
		{
			getOutputValue(i_node, i_group) /= sum;
		}
	}
}

void NeuralLayer::setFunctions(std::function<double(double)> active, std::function<double(double)> dactive)
{
	_activeFunction = active;
	_dactiveFunction = dactive;
}

void NeuralLayer::activeOutputValue()
{
	//this->weight->print();
	//prevLayer->output->print();
	d_matrix::cpyData(InputMatrix, BiasVector);
	InputMatrix->expand();
	//input->print();
	d_matrix::product(this->WeightMatrix, PrevLayer->OutputMatrix, this->InputMatrix, 1, 1);
	//this->input->print();
	for (int i = 0; i < NodeCount; i++)
	{
		for (int j = 0; j < GroupCount; j++)
		{
			OutputMatrix->getData(i, j) = _activeFunction(InputMatrix->getData(i, j));
		}
	}
}

void NeuralLayer::updateDelta()
{
	if (this->Type == Output)
	{
		d_matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
		//deltas[i] *= dactiveFunction(inputValues[i]);
		//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉
	}
	else
	{
		//nextLayer->weight->print();
		//nextLayer->delta->print();
		d_matrix::product(NextLayer->WeightMatrix, NextLayer->DeltaMatrix, DeltaMatrix, 1, 0, CblasTrans, CblasNoTrans);
		//this->delta->print();
		//这里看起来不对，常数节点没参与运算
		for (int i = 0; i < NodeCount; i++)
		{
			for (int j = 0; j < GroupCount; j++)
			{
				DeltaMatrix->getData(i, j) *= _dactiveFunction(InputMatrix->getData(i, j));
			}
		}
	}
}

void NeuralLayer::backPropagate(double learnSpeed /*= 0.5*/, double lambda /*= 0.1*/)
{
	updateDelta();
	//lambda = 0.0;
	d_matrix::product(DeltaMatrix, PrevLayer->OutputMatrix, WeightMatrix,
		learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, CblasNoTrans, CblasTrans);
	d_matrix::productVector(DeltaMatrix, _asBiasVector, BiasVector, learnSpeed / GroupCount, 1, CblasNoTrans);

}
