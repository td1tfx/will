#include "NeuralLayerResample.h"



NeuralLayerResample::NeuralLayerResample()
{
	_activeFunctionType = af_Linear;
}


NeuralLayerResample::~NeuralLayerResample()
{
	if (maxPos) delete maxPos;
}

//�����㣬����Ϊ������������Ĳ������ظ���
void NeuralLayerResample::initData2(int x1, int x2)
{
	region_m = x2;
	region_n = x1;
}

void NeuralLayerResample::resetGroupCount2()
{
	if (maxPos) delete maxPos;
	maxPos = new int[OutputCountPerGroup*GroupCount];
}

//���ӵ�ʱ�����֪������������
void NeuralLayerResample::connetPrevlayer2()
{
	ImageCountPerGroup = PrevLayer->ImageCountPerGroup;
	ImageRow = (PrevLayer->ImageRow + region_m - 1) / region_m;
	ImageCol = (PrevLayer->ImageCol + region_n - 1) / region_n;
	OutputCountPerGroup = ImageCountPerGroup*ImageRow*ImageCol;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	DeltaMatrix = new d_matrix(OutputCountPerGroup, GroupCount);
	OutputMatrix = new d_matrix(OutputCountPerGroup, GroupCount);
	maxPos = new int[OutputCountPerGroup*GroupCount];
}

void NeuralLayerResample::updateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
}

//ֱ��Ӳ��
void NeuralLayerResample::activeOutputValue()
{
	d_matrix::resample_colasImage(PrevLayer->OutputMatrix, OutputMatrix,
		PrevLayer->ImageRow, PrevLayer->ImageCol, PrevLayer->ImageCountPerGroup,
		ImageRow, ImageRow, _resampleType, &maxPos);
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//�ش���ѡ�е�λ�ã��ǳ��鷳��������Ҫһ��������¼
//ƽ��ֵģʽδ��ɣ��Ȳ�����
void NeuralLayerResample::spreadDeltaToPrevLayer()
{
	if (_resampleType == re_Findmax)
	{
		for (int i = 0; i < OutputCountPerGroup*GroupCount; i++)
		{
			PrevLayer->DeltaMatrix->getData(maxPos[i]) = DeltaMatrix->getData(i);
		}
	}
}


void NeuralLayerResample::backPropagate(double learnSpeed, double lambda)
{
	//������ûʲô������
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
