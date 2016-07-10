#include "NeuralLayerConvolution.h"



NeuralLayerConvolution::NeuralLayerConvolution()
{

}


NeuralLayerConvolution::~NeuralLayerConvolution()
{
}

void NeuralLayerConvolution::activeOutput()
{
	// 	d_matrix::convolution_colasImage(PrevLayer->OutputMatrix, UnactivedMatrix,
	// 		PrevLayer->ImageRow, PrevLayer->ImageCol, PrevLayer->ImageCount,
	// 		ImageRow, ImageRow);
	Matrix::activeForward(_activeFunctionType, XMatrix, YMatrix);
}

int NeuralLayerConvolution::saveInfo(FILE* fout)
{
	fprintf(fout, "Convolution\n%d %d %d %d\n", int(_convolutionType), kernelCount, kernelRow, kernelCol);
	kernelData->printAsVector(fout);
	return 4 + kernelCount*kernelRow*kernelCol;
}

int NeuralLayerConvolution::loadInfo(real* v, int n)
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
		kernels[i] = new Matrix(kernelRow, kernelCol,md_Outside, mc_UseCuda);
		kernels[i]->shareData(kernelData, 0, i);
	}
	return k;
}
