#include "NeuralLayerPooling.h"



NeuralLayerPooling::NeuralLayerPooling()
{
	_activeFunctionType = af_Linear;
}


NeuralLayerPooling::~NeuralLayerPooling()
{
	if (maxPos) delete maxPos;
}

//采样层，参数为本层横向和纵向的采样像素个数
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

//连接的时候才能知道本层的输出数
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

//直接硬上
void NeuralLayerPooling::activeOutputValue()
{
	Matrix::pooling(PrevLayer->OutputMatrix, OutputMatrix,
		PrevLayer->ImageRow, PrevLayer->ImageCol, PrevLayer->ImageCountPerGroup,
		ImageRow, ImageRow, _resampleType, &maxPos);
	//对于最大值采样来说，偏置、权重与激活函数均意义不大，后面再说
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//回传被选中的位置，非常麻烦，可能需要一个东西记录
//平均值模式未完成，先不管了
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
	//最大值采样没什么好练的
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
