#include "NeuralLayerFull.h"



NeuralLayerFull::NeuralLayerFull()
{
	//_activeFunctionType = af_Sigmoid;
}


NeuralLayerFull::~NeuralLayerFull()
{
	safe_delete(WeightMatrix);
	safe_delete(BiasVector);
	safe_delete(_asBiasVector);
}

//全连接层中，x1是本层输出数
void NeuralLayerFull::initData2(int x1, int x2)
{
	//deleteData();
	auto outputCount = x1;
	this->OutputCountPerGroup = outputCount;

	if (Type == lt_Input)
	{
		YMatrix = new Matrix(outputCount, GroupCount, md_Outside);
	}
	else
	{
		YMatrix = new Matrix(outputCount, GroupCount);
		XMatrix = new Matrix(outputCount, GroupCount);
	}
	if (Type == lt_Output)
	{
		ExpectMatrix = new Matrix(outputCount, GroupCount, md_Outside);
	}
	dXMatrix = new Matrix(outputCount, GroupCount);
	dXMatrix->initData(1);
	dYMatrix = new Matrix(outputCount, GroupCount);
	dYMatrix->initData(0);
	_asBiasVector = new Matrix(GroupCount, 1);
	_asBiasVector->initData(1);
	//output->print();
}

void NeuralLayerFull::resetGroupCount2()
{
	if (_asBiasVector->resize(GroupCount, 1) > 0)
	{
		_asBiasVector->initData(1);
	}
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
	Matrix::activeBackward(_activeFunctionType, YMatrix, dYMatrix, XMatrix, dXMatrix);
}

void NeuralLayerFull::activeOutput()
{
	Matrix::cpyData(XMatrix, BiasVector);
	XMatrix->expand();
	Matrix::product(this->WeightMatrix, PrevLayer->YMatrix, this->XMatrix, 1, 1);
	Matrix::activeForward(_activeFunctionType, XMatrix, YMatrix);
}

void NeuralLayerFull::spreadDeltaToPrevLayer()
{
	Matrix::product(WeightMatrix, dXMatrix, PrevLayer->dYMatrix, 1, 0, mt_Trans, mt_NoTrans);
}

void NeuralLayerFull::updateWeightBias(real learnSpeed, real lambda)
{
	Matrix::product(dXMatrix, PrevLayer->YMatrix, WeightMatrix,
		learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, mt_NoTrans, mt_Trans);
	Matrix::productVector(dXMatrix, _asBiasVector, BiasVector, learnSpeed / GroupCount, 1, mt_NoTrans);
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

