#include "NeuralLayer.h"


int NeuralLayer::GroupCount;
int NeuralLayer::EffectiveGroupCount;
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


//ÿ������ֵ���Ϊ1��������Ϊ0
void NeuralLayer::markMax()
{
	if (OutputCount <= 0) return;
	auto temp = new double[OutputCount*GroupCount];
	memset(temp, 0, sizeof(double)*OutputCount*GroupCount);
	for (int i_group = 0; i_group < GroupCount; i_group++)
	{
		int index = OutputMatrix->indexColMaxAbs(i_group);
		temp[index+i_group*OutputCount] = 1;
		//printf("%d", index);
	}
	OutputMatrix->memcpyDataIn(temp, OutputCount*GroupCount);
	delete temp;
}

//��һ�����������ʹ�ã�����㼤���Ϊexp
void NeuralLayer::normalized()
{
	for (int i_group = 0; i_group < GroupCount; i_group++)
	{
		double sum = OutputMatrix->sumColAbs(i_group);
		if (sum == 0) continue;
		OutputMatrix->colMultiply(1 / sum, i_group);
	}
}


