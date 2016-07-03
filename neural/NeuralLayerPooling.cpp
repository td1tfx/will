#include "NeuralLayerPooling.h"



NeuralLayerPooling::NeuralLayerPooling()
{
	_activeFunctionType = af_ReLU;
}


NeuralLayerPooling::~NeuralLayerPooling()
{
	if (maxPos) delete maxPos;
}

//�����㣬����Ϊ������������Ĳ������ظ���
void NeuralLayerPooling::initData2(int x1, int x2)
{
	window_w = x1;
	window_h = x2;
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
	ImageRow = (PrevLayer->ImageRow + window_w - 1) / window_w;
	ImageCol = (PrevLayer->ImageCol + window_h - 1) / window_h;
	OutputCountPerGroup = ImageCountPerGroup*ImageRow*ImageCol;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	DeltaMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	OutputMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	maxPos = new int[OutputCountPerGroup*GroupCount];
}

void NeuralLayerPooling::backPropagateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
}

//ֱ��Ӳ��
void NeuralLayerPooling::activeForwardOutput()
{
	Matrix::poolingForward(_resampleType, PrevLayer->OutputMatrix, UnactivedMatrix, window_w, window_h, w_stride, h_stride, &maxPos);
	//�������ֵ������˵��ƫ�á�Ȩ���뼤��������岻�󣬺�����˵
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//�ش���ѡ�е�λ�ã��ǳ��鷳��������Ҫһ��������¼
//ƽ��ֵģʽδ��ɣ��Ȳ�����
void NeuralLayerPooling::spreadDeltaToPrevLayer()
{
	Matrix::poolingBackward(_resampleType, OutputMatrix, DeltaMatrix, PrevLayer->OutputMatrix, PrevLayer->DeltaMatrix,
		window_w, window_h, w_stride, h_stride, maxPos);
}


void NeuralLayerPooling::updateWeightBias(double learnSpeed, double lambda)
{
	//�������ûʲô������
}

int NeuralLayerPooling::saveInfo(FILE* fout)
{
	fprintf(fout, "Resample\n%d %d %d", int(_resampleType), window_w, window_h);
	return 3;
}

int NeuralLayerPooling::loadInfo(double* v, int n)
{
	_resampleType = ResampleType(int(v[0]));
	window_w = v[1];
	window_h = v[2];
	return 3;
}
