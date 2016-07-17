#include "LayerFull.h"



LayerFull::LayerFull()
{
    //_activeFunctionType = af_Sigmoid;
}


LayerFull::~LayerFull()
{
    safe_delete(WeightMatrix);
    safe_delete(BiasVector);
    safe_delete(asBiasVector);
}

//全连接层中，x1是本层输出数
void LayerFull::initData2(LayerInitInfo* info)
{
    //deleteData();
    auto outputCount = info->full.outputCount;
    this->OutputCountPerGroup = outputCount;

    if (Type == lt_Input)
    {
        AMatrix = new Matrix(outputCount, GroupCount, md_Outside);
    }
    else
    {
        AMatrix = new Matrix(outputCount, GroupCount);
        XMatrix = new Matrix(outputCount, GroupCount);
    }
    if (Type == lt_Output)
    {
        YMatrix = new Matrix(outputCount, GroupCount, md_Outside);
    }
    dXMatrix = new Matrix(outputCount, GroupCount);
    dXMatrix->initData(1);
    dAMatrix = new Matrix(outputCount, GroupCount);
    dAMatrix->initData(0);
    asBiasVector = new Matrix(GroupCount, 1);
    asBiasVector->initData(1);
    //output->print();
}

void LayerFull::resetGroupCount2()
{
    if (asBiasVector->resize(GroupCount, 1) > 0)
    {
        asBiasVector->initData(1);
    }
}

void LayerFull::connetPrevlayer2()
{
    this->WeightMatrix = new Matrix(this->OutputCountPerGroup, PrevLayer->OutputCountPerGroup);
    this->WeightMatrix->initRandom();
    this->BiasVector = new Matrix(this->OutputCountPerGroup, 1);
    this->BiasVector->initRandom();
}

void LayerFull::activeBackward2()
{
    NextLayer->spreadDeltaToPrevLayer();
    //UnactivedMatrix->dactiveFunction(_activeFunctionType);
    //Matrix::hadamardProduct(DeltaMatrix, UnactivedMatrix, DeltaMatrix);
    Matrix::activeBackward(_activeFunctionType, AMatrix, dAMatrix, XMatrix, dXMatrix);
}

void LayerFull::activeForward()
{
    Matrix::cpyData(XMatrix, BiasVector);
    XMatrix->expand();
    Matrix::product(this->WeightMatrix, PrevLayer->AMatrix, this->XMatrix, 1, 1);
    Matrix::activeForward(_activeFunctionType, XMatrix, AMatrix);
}

void LayerFull::spreadDeltaToPrevLayer()
{
    Matrix::product(WeightMatrix, dXMatrix, PrevLayer->dAMatrix, 1, 0, Matrix_Trans, Matrix_NoTrans);
}

void LayerFull::updateParameters(real learnSpeed, real lambda)
{
    Matrix::product(dXMatrix, PrevLayer->AMatrix, WeightMatrix,
                    learnSpeed / GroupCount, 1 - lambda * learnSpeed / GroupCount, Matrix_NoTrans, Matrix_Trans);
    Matrix::productVector(dXMatrix, asBiasVector, BiasVector, learnSpeed / GroupCount, 1, Matrix_NoTrans);
}

int LayerFull::saveInfo(FILE* fout)
{
    fprintf(fout, "Full connection\n");
    fprintf(fout, "weight for layer %d to %d\n", Id, Id - 1);
    WeightMatrix->print(fout);
    fprintf(fout, "bias for layer %d\n", Id);
    BiasVector->print(fout);
    fprintf(fout, "\n");
    return 3 + WeightMatrix->getDataCount() + BiasVector->getDataCount();
}

int LayerFull::loadInfo(real* v, int n)
{
    int k = 0;
    k += 2;
    k += WeightMatrix->load(v + k, n - k);
    k += 1;
    k += BiasVector->load(v + k, n - k);
    return k;
}

