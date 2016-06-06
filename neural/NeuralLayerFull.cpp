#include "NeuralLayerFull.h"



NeuralLayerFull::NeuralLayerFull()
{
}


NeuralLayerFull::~NeuralLayerFull()
{
}

void NeuralLayerFull::initData(int nodeCount, int groupCount, NeuralLayerType type /*= Hidden*/)
{
	deleteData();
	this->Type = type;
	this->OutputCount = nodeCount;
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

void NeuralLayerFull::resetData(int groupCount)
{
	this->GroupCount = groupCount;
	InputMatrix->resize(OutputCount, groupCount);
	OutputMatrix->resize(OutputCount, groupCount);
	DeltaMatrix->resize(OutputCount, groupCount);
	ExpectMatrix->resize(OutputCount, groupCount);
	_asBiasVector->resize(groupCount, 1);
	_asBiasVector->initData(1);
}

void NeuralLayerFull::connetPrevlayer(NeuralLayer* prevLayer)
{
	int n = this->OutputCount*this->OutputCount;
	this->WeightMatrix = new d_matrix(this->OutputCount, prevLayer->OutputCount);
	this->WeightMatrix->initRandom();
	this->BiasVector = new d_matrix(this->OutputCount, 1);
	this->BiasVector->initRandom();
	this->PrevLayer = prevLayer;
	prevLayer->NextLayer = this;
}

void NeuralLayerFull::activeOutputValue()
{
	d_matrix::cpyData(InputMatrix, BiasVector);
	InputMatrix->expand();
	d_matrix::product(this->WeightMatrix, PrevLayer->OutputMatrix, this->InputMatrix, 1, 1);
	d_matrix::applyFunction(InputMatrix, OutputMatrix, _activeFunction);
}

void NeuralLayerFull::updateDelta()
{
	if (this->Type == Output)
	{
		d_matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
		//deltas[i] *= dactiveFunction(inputValues[i]);
		//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉
	}
	else
	{
		d_matrix::product(NextLayer->WeightMatrix, NextLayer->DeltaMatrix, DeltaMatrix, 1, 0, Trans, NoTrans);
		InputMatrix->applyFunction(_dactiveFunction);
		d_matrix::hadamardProduct(DeltaMatrix, InputMatrix, DeltaMatrix);
	}
}

void NeuralLayerFull::backPropagate(double learnSpeed, double lambda)
{
	updateDelta();
	//lambda = 0.0;
	d_matrix::product(DeltaMatrix, PrevLayer->OutputMatrix, WeightMatrix,
		learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, NoTrans, Trans);
	d_matrix::productVector(DeltaMatrix, _asBiasVector, BiasVector, learnSpeed / GroupCount, 1, NoTrans);
}

int NeuralLayerFull::saveInfo(FILE* fout)
{
	fprintf(fout, "weight for layer %d to %d\n", Id, Id - 1);
	WeightMatrix->print(fout);
	fprintf(fout, "bias for layer %d\n", Id);
	BiasVector->print(fout);
	fprintf(fout, "\n");
	return 3 + WeightMatrix->getDataCount() + BiasVector->getDataCount();
}

int NeuralLayerFull::readInfo(double* v, int n)
{
	int k = 0;
	k += 2;
	k += WeightMatrix->load(v + k, n - k);
	k += 1;
	k += BiasVector->load(v + k, n - k);
	return k;
}

