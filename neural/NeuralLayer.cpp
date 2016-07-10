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
	safe_delete(XMatrix);
	safe_delete(YMatrix);
	safe_delete(dXMatrix);
	safe_delete(dYMatrix);
	safe_delete(ExpectMatrix);
}

void NeuralLayer::connetPrevlayer(NeuralLayer* prevLayer)
{
	this->PrevLayer = prevLayer;
	prevLayer->NextLayer = this;
	connetPrevlayer2();
}

void NeuralLayer::resetGroupCount()
{
	XMatrix->resize(OutputCountPerGroup, GroupCount);
	YMatrix->resize(OutputCountPerGroup, GroupCount);
	dXMatrix->resize(OutputCountPerGroup, GroupCount);
	dYMatrix->resize(OutputCountPerGroup, GroupCount);
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
			Matrix::add(ExpectMatrix, -1, YMatrix, dYMatrix);
			Matrix::activeBackward(_activeFunctionType, YMatrix, dYMatrix, XMatrix, dXMatrix);
			break;
		case cf_CrossEntropy:
			if (_activeFunctionType == af_Sigmoid)
			{
				//交叉熵和Sigmoid同时使用，则有此简化方法
				Matrix::add(ExpectMatrix, -1, YMatrix, dYMatrix);
				//如果dX和dY用同一矩阵，则不需要这次复制，为通用性保留
				Matrix::cpyData(dXMatrix, dYMatrix);
			}
			else
			{
				//其余情况需自行推导
				Matrix::add(ExpectMatrix, -1, YMatrix, dYMatrix);
				Matrix::activeBackward(_activeFunctionType, YMatrix, dYMatrix, XMatrix, dXMatrix);
			}
			break;
		default:
			break;
		}
	}
	else
	{
		updateDelta2();
	}
}

