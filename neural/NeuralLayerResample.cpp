#include "NeuralLayerResample.h"



NeuralLayerResample::NeuralLayerResample()
{
	_activeFunctionType = af_Linear;
}


NeuralLayerResample::~NeuralLayerResample()
{
	if (maxPos) delete maxPos;
}

//采样层，参数为本层横向和纵向的采样像素个数
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

//连接的时候才能知道本层的输出数
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

//直接硬上
void NeuralLayerResample::activeOutputValue()
{
	d_matrix::resample_colasImage(PrevLayer->OutputMatrix, OutputMatrix,
		PrevLayer->ImageRow, PrevLayer->ImageCol, PrevLayer->ImageCountPerGroup,
		ImageRow, ImageRow, _resampleType, &maxPos);
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//回传被选中的位置，非常麻烦，可能需要一个东西记录
//平均值模式未完成，先不管了
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
	//采样层没什么好练的
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
