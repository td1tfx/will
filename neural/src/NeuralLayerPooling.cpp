#include "NeuralLayerPooling.h"



NeuralLayerPooling::NeuralLayerPooling()
{
	_activeFunctionType = af_ReLU;
}


NeuralLayerPooling::~NeuralLayerPooling()
{
	if (recordPos) delete recordPos;
}

//�����㣬����Ϊ������������Ĳ������ظ���
void NeuralLayerPooling::initData2(NeuralLayerInitInfo* info)
{
	window_w = info->window_w;
	window_h = info->window_h;
	stride_w = info->stride_w;
	stride_w = info->stride_h;
}

void NeuralLayerPooling::resetGroupCount2()
{
	if (recordPos) delete recordPos;
	recordPos = new int[OutputCountPerGroup*GroupCount];
}

//���ӵ�ʱ�����֪������������
void NeuralLayerPooling::connetPrevlayer2()
{
	ImageCountPerGroup = PrevLayer->ImageCountPerGroup;
	ImageRow = (PrevLayer->ImageRow + window_w - 1) / window_w;
	ImageCol = (PrevLayer->ImageCol + window_h - 1) / window_h;
	OutputCountPerGroup = ImageCountPerGroup*ImageRow*ImageCol;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	dAMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	AMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	recordPos = new int[OutputCountPerGroup*GroupCount];
}

void NeuralLayerPooling::updateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
}

//ֱ��Ӳ��
void NeuralLayerPooling::activeOutput()
{
	Matrix::poolingForward(_resampleType, PrevLayer->AMatrix, XMatrix, window_w, window_h, stride_w, stride_h, recordPos);
	//�������ֵ������˵��ƫ�á�Ȩ���뼤��������岻�󣬺�����˵
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//�ش���ѡ�е�λ�ã��ǳ��鷳��������Ҫһ��������¼
//ƽ��ֵģʽδ��ɣ��Ȳ�����
void NeuralLayerPooling::spreadDeltaToPrevLayer()
{
	Matrix::poolingBackward(_resampleType, AMatrix, dAMatrix, PrevLayer->AMatrix, PrevLayer->dAMatrix,
		window_w, window_h, stride_w, stride_h, recordPos);
}


void NeuralLayerPooling::updateWeightBias(real learnSpeed, real lambda)
{
	//�������ûʲô������
}

int NeuralLayerPooling::saveInfo(FILE* fout)
{
	fprintf(fout, "Resample\n%d %d %d", int(_resampleType), window_w, window_h);
	return 3;
}

int NeuralLayerPooling::loadInfo(real* v, int n)
{
	_resampleType = ResampleType(int(v[0]));
	window_w = v[1];
	window_h = v[2];
	return 3;
}