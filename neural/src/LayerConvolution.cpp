#include "LayerConvolution.h"

LayerConvolution::LayerConvolution()
{
}

LayerConvolution::~LayerConvolution()
{
}

void LayerConvolution::activeForward()
{
    Matrix::activeForward(_activeFunctionType, XMatrix, AMatrix);
}

int LayerConvolution::saveInfo(FILE* fout)
{
    fprintf(fout, "Convolution\n%d %d %d %d\n", int(_convolutionType), kernelCount, kernelRow, kernelCol);
    kernelData->printAsVector(fout);
    return 4 + kernelCount * kernelRow * kernelCol;
}

int LayerConvolution::loadInfo(real* v, int n)
{
    int k = 0;
    _convolutionType = ConvolutionType(int(v[k++]));
    kernelCount = v[k++];
    kernelRow = v[k++];
    kernelCol = v[k++];
    kernelData = new Matrix(kernelRow * kernelCol, kernelCount);
    k += kernelData->loadAsVector(v + k, n - k);
    for (int i = 0; i < kernelCount; i++)
    {
        kernels[i] = new Matrix(kernelRow, kernelCol, md_Outside, mc_UseCuda);
        kernels[i]->shareData(kernelData, 0, i);
    }
    return k;
}
