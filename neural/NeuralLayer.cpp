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
			d_matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
			UnactivedMatrix->dactiveFunction(_activeFunctionType);
			d_matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
			break;
		case cf_CrossEntropy:
			d_matrix::minus(ExpectMatrix, OutputMatrix, DeltaMatrix);
			break;
		default:
			break;
		}
		//�������ȥ������˷�����ʹ�ý�������Ϊ���ۺ��������������ز�Ĵ���������ȥ�������巽���Լ��Ƶ���
	}
	else
	{
		updateDelta2();
	}
}

