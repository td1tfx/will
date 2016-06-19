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
	d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

int NeuralLayerConvolution::saveInfo(FILE* fout)
{
	int k = 2;
	fprintf(fout, "Convolution\n%d %d", int(_convolutionType), _kernelCount);
	for (int i = 0; i < _kernelCount; i++)
	{
		fprintf(fout, "%d, %d\n", kernels[i]->getRow(), kernels[i]->getCol());
		kernels[i]->print(fout);
		k += 2 + kernels[i]->getRow()*kernels[i]->getCol();
	}
	return k;
}

int NeuralLayerConvolution::loadInfo(double* v, int n)
{
	int k = 0;
	_convolutionType = ConvolutionType(int(v[k++]));
	_kernelCount = v[k++];
	for (int i = 0; i < _kernelCount; i++)
	{
		int m = v[k++];
		int n = v[k++];
		kernels[i] = new d_matrix(m, n, 1, 1);
		k += kernels[i]->load(&v[k], n - k);
	}
	return k;
}
