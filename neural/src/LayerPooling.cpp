#include "LayerPooling.h"



LayerPooling::LayerPooling()
{
    _activeFunctionType = af_ReLU;
}


LayerPooling::~LayerPooling()
{
    if (recordPos) { delete recordPos; }
}

//�����㣬����Ϊ������������Ĳ������ظ���
void LayerPooling::initData2(LayerInitInfo* info)
{
    window_w = info->pooling.window_w;
    window_h = info->pooling.window_h;
    stride_w = info->pooling.stride_w;
    stride_w = info->pooling.stride_h;
}

void LayerPooling::resetGroupCount2()
{
    if (recordPos) { delete recordPos; }
    recordPos = new int[OutputCountPerGroup * GroupCount];
}

//���ӵ�ʱ�����֪������������
void LayerPooling::connetPrevlayer2()
{
    ImageCountPerGroup = PrevLayer->ImageCountPerGroup;
    ImageRow = (PrevLayer->ImageRow + window_w - 1) / window_w;
    ImageCol = (PrevLayer->ImageCol + window_h - 1) / window_h;
    OutputCountPerGroup = ImageCountPerGroup * ImageRow * ImageCol;
    //UnactivedMatrix = new d_matrix(OutputCount, GroupCount);
    dAMatrix = new Matrix(OutputCountPerGroup, GroupCount);
    AMatrix = new Matrix(OutputCountPerGroup, GroupCount);
    recordPos = new int[OutputCountPerGroup * GroupCount];
}

void LayerPooling::activeBackward2()
{
    NextLayer->spreadDeltaToPrevLayer();
}

//ֱ��Ӳ��
void LayerPooling::activeForward()
{
    Matrix::poolingForward(_resampleType, PrevLayer->AMatrix, XMatrix, window_w, window_h, stride_w, stride_h, recordPos);
    //�������ֵ������˵��ƫ�á�Ȩ���뼤��������岻�󣬺�����˵
    //d_matrix::activeFunction(UnactivedMatrix, OutputMatrix, _activeFunctionType);
}

//�ش���ѡ�е�λ�ã��ǳ��鷳��������Ҫһ��������¼
//ƽ��ֵģʽδ��ɣ��Ȳ�����
void LayerPooling::spreadDeltaToPrevLayer()
{
    Matrix::poolingBackward(_resampleType, AMatrix, dAMatrix, PrevLayer->AMatrix, PrevLayer->dAMatrix,
                            window_w, window_h, stride_w, stride_h, recordPos);
}


void LayerPooling::updateParameters(real learnSpeed, real lambda)
{
    //�������ûʲô������
}

int LayerPooling::saveInfo(FILE* fout)
{
    fprintf(fout, "Resample\n%d %d %d", int(_resampleType), window_w, window_h);
    return 3;
}

int LayerPooling::loadInfo(real* v, int n)
{
    _resampleType = PoolingType(int(v[0]));
    window_w = v[1];
    window_h = v[2];
    return 3;
}
