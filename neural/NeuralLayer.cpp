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
		//���ۺ��������������
		switch (_costFunctionType)
		{
		case cf_RMSE:
			Matrix::add(ExpectMatrix, -1, YMatrix, dYMatrix);
			Matrix::activeBackward(_activeFunctionType, YMatrix, dYMatrix, XMatrix, dXMatrix);
			break;
		case cf_CrossEntropy:
			if (_activeFunctionType == af_Sigmoid)
			{
				//�����غ�Sigmoidͬʱʹ�ã����д˼򻯷���
				Matrix::add(ExpectMatrix, -1, YMatrix, dYMatrix);
				//���dX��dY��ͬһ��������Ҫ��θ��ƣ�Ϊͨ���Ա���
				Matrix::cpyData(dXMatrix, dYMatrix);
			}
			else
			{
				//��������������Ƶ�
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

