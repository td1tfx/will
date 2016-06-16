#include "NeuralLayerFull.h"



NeuralLayerFull::NeuralLayerFull()
{
}


NeuralLayerFull::~NeuralLayerFull()
{
	if (WeightMatrix) { delete WeightMatrix; }
	if (BiasVector) { delete BiasVector; }
	if (_asBiasVector) { delete _asBiasVector; }
}

//全连接层中，x1是本层输出数
void NeuralLayerFull::initData(NeuralLayerType type, int x1, int x2)
{
	//deleteData();
	this->Type = type;
	this->OutputCount = x1;

	if (type == Input)
	{
		OutputMatrix = new d_matrix(x1, GroupCount, 0);
	}
	else
	{
		OutputMatrix = new d_matrix(x1, GroupCount);
		UnactivedMatrix = new d_matrix(x1, GroupCount);
	}
	if (type == Output)
	{
		ExpectMatrix = new d_matrix(x1, GroupCount, 0);
	}

	DeltaMatrix = new d_matrix(x1, GroupCount);
	_asBiasVector = new d_matrix(GroupCount, 1);
	_asBiasVector->initData(1);
	//output->print();
}

void NeuralLayerFull::resetGroupCount()
{
	UnactivedMatrix->resize(OutputCount, GroupCount);
	OutputMatrix->resize(OutputCount, GroupCount);
	DeltaMatrix->resize(OutputCount, GroupCount);
	ExpectMatrix->resize(OutputCount, GroupCount);
	if (_asBiasVector->resize(GroupCount, 1) > 0)
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
	d_matrix::cpyData(UnactivedMatrix, BiasVector);
	UnactivedMatrix->expand();
	d_matrix::product(this->WeightMatrix, PrevLayer->OutputMatrix, this->UnactivedMatrix, 1, 1);
	//d_matrix::productVector2(this->WeightMatrix, PrevLayer->OutputMatrix, this->InputMatrix, 1, 1);
	d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, ActiveMode);
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
		NextLayer->spreadDeltaToPrevLayer();
		UnactivedMatrix->dactiveFunction(ActiveMode);
		d_matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
	}
}

void NeuralLayerFull::spreadDeltaToPrevLayer()
{
	d_matrix::product(WeightMatrix, DeltaMatrix, PrevLayer->DeltaMatrix, 1, 0, Trans, NoTrans);
}

void NeuralLayerFull::backPropagate(double learnSpeed, double lambda)
{
	updateDelta();
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

int NeuralLayerFull::loadInfo(double* v, int n)
{
	int k = 0;
	k += 2;
	k += WeightMatrix->load(v + k, n - k);
	k += 1;
	k += BiasVector->load(v + k, n - k);
	return k;
}

