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

void NeuralLayer::resetData(int groupCount)
{
	initData(this->OutputCount, groupCount, this->Type);
}


//每组的最大值标记为1，其余标记为0
void NeuralLayer::markMax()
{
	if (OutputCount <= 0) return;
	for (int i_group = 0; i_group < GroupCount; i_group++)
	{
		int index = OutputMatrix->indexColMaxAbs(i_group);
		for (int i_node = 0; i_node < OutputCount; i_node++)
		{
			OutputMatrix->getData(i_node, i_group) = 0;
		}
		OutputMatrix->getData(index, i_group) = 1;
	}
}

//归一化，计算概率使用，输出层激活函数为exp
void NeuralLayer::normalized()
{
	for (int i_group = 0; i_group < GroupCount; i_group++)
	{
		double sum = OutputMatrix->sumColAbs(i_group);
		if (sum == 0) continue;
		OutputMatrix->colMultiply(1 / sum, i_group);
	}
}

void NeuralLayer::setFunctions(std::function<double(double)> active, std::function<double(double)> dactive)
{
	_activeFunction = active;
	_dactiveFunction = dactive;
}

