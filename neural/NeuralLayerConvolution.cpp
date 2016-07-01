#include "NeuralLayerConvolution.h"



NeuralLayerConvolution::NeuralLayerConvolution()
{

}


NeuralLayerConvolution::~NeuralLayerConvolution()
{
}

void NeuralLayerConvolution::activeOutputValue()
{
	// 	d_matrix::convolution_colasImage(PrevLayer->OutputMatrix, UnactivedMatrix,
	// 		PrevLayer->ImageRow, PrevLayer->ImageCol, PrevLayer->ImageCount,
	// 		ImageRow, ImageRow);
	Matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

int NeuralLayerConvolution::saveInfo(FILE* fout)
{
	fprintf(fout, "Convolution\n%d %d %d %d\n", int(_convolutionType), kernelCount, kernelRow, kernelCol);
	kernelData->printAsVector(fout);
	return 4 + kernelCount*kernelRow*kernelCol;
}

int NeuralLayerConvolution::loadInfo(double* v, int n)
{
	int k = 0;
	_convolutionType = ConvolutionType(int(v[k++]));
	kernelCount = v[k++];
	kernelRow = v[k++];
	kernelCol = v[k++];
	kernelData = new Matrix(kernelRow*kernelCol, kernelCount);
	k += kernelData->loadAsVector(v + k, n - k);
	for (int i = 0; i < kernelCount; i++)
	{
		kernels[i] = new Matrix(kernelRow, kernelCol, 0, 1);
		kernels[i]->shareData(kernelData, 0, i);
	}
	return k;
}
