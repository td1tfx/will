#include "NeuralLayerFull.h"



NeuralLayerFull::NeuralLayerFull()
{
	//_activeFunctionType = af_Sigmoid;
}


NeuralLayerFull::~NeuralLayerFull()
{
	safe_delete(WeightMatrix);
	safe_delete(BiasVector);
	safe_delete(asBiasVector);
}

//全连接层中，x1是本层输出数
void NeuralLayerFull::initData2(NeuralLayerInitInfo* info)
{
	//deleteData();
	auto outputCount = info->full.outputCount;
	this->OutputCountPerGroup = outputCount;

	if (Type == lt_Input)
	{
		AMatrix = new Matrix(outputCount, GroupCount, md_Outside);
	}
	else
	{
		AMatrix = new Matrix(outputCount, GroupCount);
		XMatrix = new Matrix(outputCount, GroupCount);
	}
	if (Type == lt_Output)
	{
		YMatrix = new Matrix(outputCount, GroupCount, md_Outside);
	}
	dXMatrix = new Matrix(outputCount, GroupCount);
	dXMatrix->initData(1);
	dAMatrix = new Matrix(outputCount, GroupCount);
	dAMatrix->initData(0);
	asBiasVector = new Matrix(GroupCount, 1);
	asBiasVector->initData(1);
	//output->print();
}

void NeuralLayerFull::resetGroupCount2()
{
	if (asBiasVector->resize(GroupCount, 1) > 0)
	{
		asBiasVector->initData(1);
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
	Matrix::activeBackward(_activeFunctionType, AMatrix, dAMatrix, XMatrix, dXMatrix);
}

void NeuralLayerFull::activeOutput()
{
	Matrix::cpyData(XMatrix, BiasVector);
	XMatrix->expand();
	Matrix::product(this->WeightMatrix, PrevLayer->AMatrix, this->XMatrix, 1, 1);
	Matrix::activeForward(_activeFunctionType, XMatrix, AMatrix);
}

void NeuralLayerFull::spreadDeltaToPrevLayer()
{
	Matrix::product(WeightMatrix, dXMatrix, PrevLayer->dAMatrix, 1, 0, mt_Trans, mt_NoTrans);
}

void NeuralLayerFull::updateWeightBias(real learnSpeed, real lambda)
{
	Matrix::product(dXMatrix, PrevLayer->AMatrix, WeightMatrix,
		learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, mt_NoTrans, mt_Trans);
	Matrix::productVector(dXMatrix, asBiasVector, BiasVector, learnSpeed / GroupCount, 1, mt_NoTrans);
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

