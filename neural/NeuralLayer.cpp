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
	ImageCountPerGroup = count;
	if (count <= 0)
		ImageCountPerGroup = OutputCountPerGroup / w / h;
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
	UnactivedMatrix->resize(OutputCountPerGroup, GroupCount);
	OutputMatrix->resize(OutputCountPerGroup, GroupCount);
	DeltaMatrix->resize(OutputCountPerGroup, GroupCount);
	ExpectMatrix->resize(OutputCountPerGroup, GroupCount);
	resetGroupCount2();
}

void NeuralLayer::updateDelta()
{
	if (this->Type == lt_Output)
	{
		//代价函数由这里决定！
		switch (_costFunctionType)
		{
		case cf_RMSE:
			Matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
			//UnactivedMatrix->activeBackward(_activeFunctionType);
			//Matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
			Matrix::activeBackward(_activeFunctionType, UnactivedMatrix, OutputMatrix, DeltaMatrix);
			break;
		case cf_CrossEntropy:
			Matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
			break;
		default:
			break;
		}
		//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉！具体方程自己推导！
	}
	else
	{
		updateDelta2();
	}
}

