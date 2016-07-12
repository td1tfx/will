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

//连接的时候才能知道本层的输出数
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

//直接硬上
void NeuralLayerPooling::activeOutput()
{
	Matrix::poolingForward(_resampleType, PrevLayer->AMatrix, XMatrix, window_w, window_h, stride_w, stride_h, recordPos);
	//对于最大值采样来说，偏置、权重与激活函数均意义不大，后面再说
	//d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//回传被选中的位置，非常麻烦，可能需要一个东西记录
//平均值模式未完成，先不管了
void NeuralLayerPooling::spreadDeltaToPrevLayer()
{
	Matrix::poolingBackward(_resampleType, AMatrix, dAMatrix, PrevLayer->AMatrix, PrevLayer->dAMatrix,
		window_w, window_h, stride_w, stride_h, recordPos);
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
