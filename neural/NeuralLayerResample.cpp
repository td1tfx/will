#include "NeuralLayerResample.h"



NeuralLayerResample::NeuralLayerResample()
{
	_activeFunctionType = af_Linear;
}


NeuralLayerResample::~NeuralLayerResample()
{
}

//采样层，参数为本层横向和纵向的采样像素个数
void NeuralLayerResample::initData2(int x1, int x2)
{
	region_m = x2;
	region_n = x1;
}

//连接的时候才能知道本层的输出数
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

//直接硬上
void NeuralLayerResample::activeOutputValue()
{
	d_matrix::resample_colasImage(PrevLayer->OutputMatrix, UnactivedMatrix, 
		PrevLayer->ImageRow, PrevLayer->ImageCol,PrevLayer->ImageCount,
		ImageRow, ImageRow, _resampleType);
	d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//回传被选中的位置，非常麻烦，可能需要一个东西记录
void NeuralLayerResample::spreadDeltaToPrevLayer()
{
	//这里要仔细推导一下
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
