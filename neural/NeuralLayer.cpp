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
		//���ۺ��������������
		switch (_costFunctionType)
		{
		case cf_RMSE:
			Matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
			Matrix::activeBackward(_activeFunctionType, OutputMatrix, UnactivedMatrix, DeltaMatrix);
			break;
		case cf_CrossEntropy:
			if (_activeFunctionType == af_Sigmoid)
			{
				//�����غ�Sigmoidͬʱʹ�ã����д˼򻯷���
				Matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
			}
			else
			{
				//��������������Ƶ�
				Matrix::activeBackward(_activeFunctionType, OutputMatrix, UnactivedMatrix, DeltaMatrix);
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

