#include "NeuralLayer.h"


int NeuralLayer::GroupCount;
int NeuralLayer::Step;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	deleteData();
}


void NeuralLayer::setImageMode(int w, int h, int count)
{
	ImageRow = h; 
	ImageCol = w; 
	ImageCount = count;
	if (count <= 0)
		ImageCount = OutputCount / w / h;
}

void NeuralLayer::deleteData()
{
	if (UnactivedMatrix) { delete UnactivedMatrix; }
	if (OutputMatrix) { delete OutputMatrix; }
	if (DeltaMatrix) { delete DeltaMatrix; }
	if (ExpectMatrix) { delete ExpectMatrix; }
}

void NeuralLayer::connetPrevlayer(NeuralLayer* prevLayer)
{
	this->PrevLayer = prevLayer;
	prevLayer->NextLayer = this;
	connetPrevlayer2();
}

void NeuralLayer::resetGroupCount()
{
	UnactivedMatrix->resize(OutputCount, GroupCount);
	OutputMatrix->resize(OutputCount, GroupCount);
	DeltaMatrix->resize(OutputCount, GroupCount);
	ExpectMatrix->resize(OutputCount, GroupCount);
	resetGroupCount2();
}

void NeuralLayer::updateDelta()
{
	if (this->Type == lt_Output)
	{
		//代价函数由这里决定！
		d_matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
		//deltas[i] *= dactiveFunction(inputValues[i]);
		//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉！具体方程自己推导！
	}
	else
	{
		NextLayer->spreadDeltaToPrevLayer();
		UnactivedMatrix->dactiveFunction(_activeFunctionType);
		d_matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
		//updateDelta2();
	}
}

