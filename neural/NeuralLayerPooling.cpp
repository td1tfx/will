#include "NeuralLayerPooling.h"



NeuralLayerPooling::NeuralLayerPooling()
{
	_activeFunctionType = af_ReLU;
}


NeuralLayerPooling::~NeuralLayerPooling()
{
	if (recordPos) delete recordPos;
}

//采样层，参数为本层横向和纵向的采样像素个数
void NeuralLayerPooling::initData2(int x1, int x2)
{
	window_w = x1;
	window_h = x2;
}

void NeuralLayerPooling::resetGroupCount2()
{
	if (recordPos) delete recordPos;
	recordPos = new int[OutputCountPerGroup*GroupCount];
}

//连接的时候才能知道本层的输出数
void NeuralLayerPooling::connetPrevlayer2()
{
	ImageCountPerGroup = PrevLayer->ImageCountPerGroup;
	ImageRow = (PrevLayer->ImageRow + window_w - 1) / window_w;
	ImageCol = (PrevLayer->ImageCol + window_h - 1) / window_h;
	OutputCountPerGroup = ImageCountPerGroup*ImageRow*ImageCol;
	//UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
	dYMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	YMatrix = new Matrix(OutputCountPerGroup, GroupCount);
	recordPos = new int[OutputCountPerGroup*GroupCount];
}

void NeuralLayerPooling::updateDelta2()
{
	NextLayer->spreadDeltaToPrevLayer();
}

//直接硬上
void NeuralLayerPooling::activeOutput()
{
	Matrix::poolingForward(_resampleType, PrevLayer->YMatrix, XMatrix, window_w, window_h, w_stride, h_stride, recordPos);
	//对于最大值采样来说，偏置、权重与激活函数均意义不大，后面再说
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//回传被选中的位置，非常麻烦，可能需要一个东西记录
//平均值模式未完成，先不管了
void NeuralLayerPooling::spreadDeltaToPrevLayer()
{
	Matrix::poolingBackward(_resampleType, YMatrix, dYMatrix, PrevLayer->YMatrix, PrevLayer->dYMatrix,
		window_w, window_h, w_stride, h_stride, recordPos);
}


void NeuralLayerPooling::updateWeightBias(real learnSpeed, real lambda)
{
	//这里好像没什么好练的
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
