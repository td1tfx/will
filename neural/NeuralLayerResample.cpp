#include "NeuralLayerResample.h"



NeuralLayerResample::NeuralLayerResample()
{
}


NeuralLayerResample::~NeuralLayerResample()
{
}

//采样层，参数为本层横向和纵向的采样像素个数
void NeuralLayerResample::initData2(int x1, int x2)
{
	scale_m = x2;
	scale_n = x1;
}

//连接的时候才能知道本层的输出数
void NeuralLayerResample::connetPrevlayer2()
{
	ImageCount = PrevLayer->ImageCount;
	ImageRow = (PrevLayer->ImageRow + scale_m - 1) / scale_m;
	ImageCol = (PrevLayer->ImageCol + scale_n - 1) / scale_n;
	OutputCount = ImageCount*ImageRow*ImageCount;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	DeltaMatrix = new d_matrix(OutputCount, GroupCount);
	OutputMatrix = new d_matrix(OutputCount, GroupCount);
}

//直接硬上
void NeuralLayerResample::activeOutputValue()
{
	auto image = new d_matrix(ImageRow, ImageCol, 0, 1);
	auto imageP = new d_matrix(PrevLayer->ImageRow, ImageCol, 0, 1);
	d_matrix::resample();
}
