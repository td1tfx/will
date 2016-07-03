#include "NeuralLayerPooling.h"



NeuralLayerPooling::NeuralLayerPooling()
{
	_activeFunctionType = af_Linear;
}


NeuralLayerPooling::~NeuralLayerPooling()
{
	if (maxPos) delete maxPos;
}

//�����㣬����Ϊ������������Ĳ������ظ���
void NeuralLayerPooling::initData2(int x1, int x2)
{
	region_m = x2;
	region_n = x1;
}

void NeuralLayerPooling::resetGroupCount2()
{
	if (maxPos) delete maxPos;
	maxPos = new int[OutputCountPerGroup*GroupCount];
}

//���ӵ�ʱ�����֪������������
void NeuralLayerPooling::connetPrevlayer2()
{
	ImageCountPerGroup = PrevLayer->ImageCountPerGroup;
	ImageRow = (PrevLayer->ImageRow + region_m - 1) / region_m;
	ImageCol = (PrevLayer->ImageCol + region_n - 1) / region_n;
	OutputCountPerGroup = ImageCountPerGroup*ImageRow*ImageCol;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	DeltaMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	OutputMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	maxPos = new int[OutputCountPerGroup*GroupCount];
}

void NeuralLayerPooling::updateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
}

//ֱ��Ӳ��
void NeuralLayerPooling::activeOutputValue()
{
	Matrix::pooling(PrevLayer->OutputMatrix, OutputMatrix,
		PrevLayer->ImageRow, PrevLayer->ImageCol, PrevLayer->ImageCountPerGroup,
		ImageRow, ImageRow, _resampleType, &maxPos);
	//�������ֵ������˵��ƫ�á�Ȩ���뼤��������岻�󣬺�����˵
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//�ش���ѡ�е�λ�ã��ǳ��鷳��������Ҫһ��������¼
//ƽ��ֵģʽδ��ɣ��Ȳ�����
void NeuralLayerPooling::spreadDeltaToPrevLayer()
{
	if (_resampleType == re_Max)
	{
		for (int i = 0; i < OutputCountPerGroup*GroupCount; i++)
		{
			PrevLayer->DeltaMatrix->getData(maxPos[i]) = DeltaMatrix->getData(i);
		}
	}
}


void NeuralLayerPooling::backPropagate(double learnSpeed, double lambda)
{
	//���ֵ����ûʲô������
}

int NeuralLayerPooling::saveInfo(FILE* fout)
{
	fprintf(fout, "Resample\n%d %d %d", int(_resampleType), region_m, region_n);
	return 3;
}

int NeuralLayerPooling::loadInfo(double* v, int n)
{
	_resampleType = ResampleType(int(v[0]));
	region_m = v[1];
	region_n = v[2];
	return 3;
}
