#include "NeuralLayerFull.h"



NeuralLayerFull::NeuralLayerFull()
{
	//_activeFunctionType = af_Sigmoid;
}


NeuralLayerFull::~NeuralLayerFull()
{
	if (WeightMatrix) { delete WeightMatrix; }
	if (BiasVector) { delete BiasVector; }
	if (_asBiasVector) { delete _asBiasVector; }
}

//全连接层中，x1是本层输出数
void NeuralLayerFull::initData2(int x1, int x2)
{
	//deleteData();
	this->OutputCountPerGroup = x1;

	if (Type == lt_Input)
	{
		OutputMatrix = new Matrix(x1, GroupCount, md_Outside);
	}
	else
	{
		OutputMatrix = new Matrix(x1, GroupCount);
		UnactivedMatrix = new Matrix(x1, GroupCount);
	}
	if (Type == lt_Output)
	{
		ExpectMatrix = new Matrix(x1, GroupCount, md_Outside);
	}

	DeltaMatrix = new Matrix(x1, GroupCount);
	_asBiasVector = new Matrix(GroupCount, 1);
	_asBiasVector->initData(1);
	//output->print();
}

void NeuralLayerFull::resetGroupCount2()
{
	if (_asBiasVector->resize(GroupCount, 1) > 0)
		_asBiasVector->initData(1);
}

void NeuralLayerFull::connetPrevlayer2()
{
	this->WeightMatrix = new Matrix(this->OutputCountPerGroup, PrevLayer->OutputCountPerGroup);
	this->WeightMatrix->initRandom();
	this->BiasVector = new Matrix(this->OutputCountPerGroup, 1);
	this->BiasVector->initRandom();
}

void NeuralLayerFull::updateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
	//UnactivedMatrix->dactiveFunction(_activeFunctionType);
	//Matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
	Matrix::activeBackward(_activeFunctionType, OutputMatrix, UnactivedMatrix, DeltaMatrix);
}

void NeuralLayerFull::activeOutput()
{
	Matrix::cpyData(UnactivedMatrix, BiasVector);
	UnactivedMatrix->expand();
	Matrix::product(this->WeightMatrix, PrevLayer->OutputMatrix, this->UnactivedMatrix, 1, 1);
	//d_matrix::productVector2(this->WeightMatrix, PrevLayer->OutputMatrix, this->InputMatrix, 1, 1);
	Matrix::activeForward(_activeFunctionType, UnactivedMatrix, OutputMatrix);
}

void NeuralLayerFull::spreadDeltaToPrevLayer()
{
	Matrix::product(WeightMatrix, DeltaMatrix, PrevLayer->DeltaMatrix, 1, 0, mt_Trans, mt_NoTrans);
}

void NeuralLayerFull::updateWeightBias(real learnSpeed, real lambda)
{
	Matrix::product(DeltaMatrix, PrevLayer->OutputMatrix, WeightMatrix,
		learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, mt_NoTrans, mt_Trans);
	Matrix::productVector(DeltaMatrix, _asBiasVector, BiasVector, learnSpeed / GroupCount, 1, mt_NoTrans);
}

int NeuralLayerFull::saveInfo(FILE* fout)
{
	fprintf(fout, "Full connection\n");
	fprintf(fout, "weight for layer %d to %d\n", Id, Id - 1);
	WeightMatrix->print(fout);
	fprintf(fout, "bias for layer %d\n", Id);
	BiasVector->print(fout);
	fprintf(fout, "\n");
	return 3 + WeightMatrix->getDataCount() + BiasVector->getDataCount();
}

int NeuralLayerFull::loadInfo(real* v, int n)
{
	int k = 0;
	k += 2;
	k += WeightMatrix->load(v + k, n - k);
	k += 1;
	k += BiasVector->load(v + k, n - k);
	return k;
}

