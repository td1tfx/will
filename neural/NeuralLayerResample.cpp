#include "NeuralLayerResample.h"



NeuralLayerResample::NeuralLayerResample()
{
	_activeFunctionType = af_Linear;
}


NeuralLayerResample::~NeuralLayerResample()
{
}

//�����㣬����Ϊ������������Ĳ������ظ���
void NeuralLayerResample::initData2(int x1, int x2)
{
	region_m = x2;
	region_n = x1;
}

//���ӵ�ʱ�����֪������������
void NeuralLayerResample::connetPrevlayer2()
{
	ImageCount = PrevLayer->ImageCount;
	ImageRow = (PrevLayer->ImageRow + region_m - 1) / region_m;
	ImageCol = (PrevLayer->ImageCol + region_n - 1) / region_n;
	OutputCount = ImageCount*ImageRow*ImageCount;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	DeltaMatrix = new d_matrix(OutputCount, GroupCount);
	OutputMatrix = new d_matrix(OutputCount, GroupCount);
}

//ֱ��Ӳ��
void NeuralLayerResample::activeOutputValue()
{
	d_matrix::resample_colasImage(PrevLayer->OutputMatrix, UnactivedMatrix, 
		PrevLayer->ImageRow, PrevLayer->ImageCol,PrevLayer->ImageCount,
		ImageRow, ImageRow, _resampleType);
	d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//�ش���ѡ�е�λ�ã��ǳ��鷳��������Ҫһ��������¼
void NeuralLayerResample::spreadDeltaToPrevLayer()
{
	//����Ҫ��ϸ�Ƶ�һ��
}


int NeuralLayerResample::saveInfo(FILE* fout)
{
	fprintf(fout, "Resample\n%d %d %d", int(_resampleType), region_m, region_n);
	return 3;
}

int NeuralLayerResample::loadInfo(double* v, int n)
{
	_resampleType = ResampleType(int(v[0])); 
	region_m = v[1];
	region_n = v[2];
	return 3;
}
