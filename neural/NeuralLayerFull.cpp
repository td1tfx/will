#include "NeuralLayerFull.h"



NeuralLayerFull::NeuralLayerFull()
{
	_activeFunctionType = af_Sigmoid;
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
		OutputMatrix = new d_matrix(x1, GroupCount, 0);
	}
	else
	{
		OutputMatrix = new d_matrix(x1, GroupCount);
		UnactivedMatrix = new d_matrix(x1, GroupCount);
	}
	if (Type == lt_Output)
	{
		ExpectMatrix = new d_matrix(x1, GroupCount, 0);
	}

	DeltaMatrix = new d_matrix(x1, GroupCount);
	_asBiasVector = new d_matrix(GroupCount, 1);
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
	this->WeightMatrix = new d_matrix(this->OutputCountPerGroup, PrevLayer->OutputCountPerGroup);
	this->WeightMatrix->initRandom();
	this->BiasVector = new d_matrix(this->OutputCountPerGroup, 1);
	this->BiasVector->initRandom();
}

void NeuralLayerFull::updateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
	UnactivedMatrix->dactiveFunction(_activeFunctionType);
	d_matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
}

void NeuralLayerFull::activeOutputValue()
{
	d_matrix::cpyData(UnactivedMatrix, BiasVector);
	UnactivedMatrix->expand();
	d_matrix::product(this->WeightMatrix, PrevLayer->OutputMatrix, this->UnactivedMatrix, 1, 1);
	//d_matrix::productVector2(this->WeightMatrix, PrevLayer->OutputMatrix, this->InputMatrix, 1, 1);
	d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

void NeuralLayerFull::spreadDeltaToPrevLayer()
{
	d_matrix::product(WeightMatrix, DeltaMatrix, PrevLayer->DeltaMatrix, 1, 0, mt_Trans, mt_NoTrans);
}

void NeuralLayerFull::backPropagate(double learnSpeed, double lambda)
{
	d_matrix::product(DeltaMatrix, PrevLayer->OutputMatrix, WeightMatrix,
		learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, mt_NoTrans, mt_Trans);
	d_matrix::productVector(DeltaMatrix, _asBiasVector, BiasVector, learnSpeed / GroupCount, 1, mt_NoTrans);
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

int NeuralLayerFull::loadInfo(double* v, int n)
{
	int k = 0;
	k += 2;
	k += WeightMatrix->load(v + k, n - k);
	k += 1;
	k += BiasVector->load(v + k, n - k);
	return k;
}

