#include "Layer.h"

int Layer::GroupCount;
int Layer::Step;

Layer::Layer()
{
}


Layer::~Layer()
{
    deleteData();
    //fprintf(stderr, "~Layer.\n");
}


void Layer::setImageMode(int w, int h, int count)
{
    ImageRow = h;
    ImageCol = w;
    ImageCountPerGroup = count;
    if (count <= 0)
    { ImageCountPerGroup = OutputCountPerGroup / w / h; }
}

void Layer::deleteData()
{
    safe_delete(XMatrix);
    safe_delete(AMatrix);
    safe_delete(dXMatrix);
    safe_delete(dAMatrix);
    safe_delete(YMatrix);
}

void Layer::connetPrevlayer(Layer* prevLayer)
{
    this->PrevLayer = prevLayer;
    prevLayer->NextLayer = this;
    connetPrevlayer2();
}

void Layer::resetGroupCount()
{
    XMatrix->resize(OutputCountPerGroup, GroupCount);
    AMatrix->resize(OutputCountPerGroup, GroupCount);
    dXMatrix->resize(OutputCountPerGroup, GroupCount);
    dAMatrix->resize(OutputCountPerGroup, GroupCount);
    YMatrix->resize(OutputCountPerGroup, GroupCount);
    resetGroupCount2();
}

void Layer::activeBackward()
{
    if (this->Type == lt_Output)
    {
        //实际上代价函数的导数形式不必拘泥于具体的推导，用这个就够了
        Matrix::add(YMatrix, -1, AMatrix, dAMatrix);
        Matrix::cpyData(dXMatrix, dAMatrix);
        return;
        //代价函数由这里决定！
        switch (_costFunctionType)
        {
        case cf_RMSE:
            Matrix::add(YMatrix, -1, AMatrix, dAMatrix);
            Matrix::activeBackward(_activeFunctionType, AMatrix, dAMatrix, XMatrix, dXMatrix);
            break;
        case cf_CrossEntropy:
            if (_activeFunctionType == af_Sigmoid)
            {
                //交叉熵和Sigmoid同时使用，则有此简化方法
                Matrix::add(YMatrix, -1, AMatrix, dAMatrix);
                //如果dX和dY用同一矩阵，则不需要这次复制，为通用性保留
                Matrix::cpyData(dXMatrix, dAMatrix);
            }
            else
            {
                //其余情况需自行推导
                Matrix::add(YMatrix, -1, AMatrix, dAMatrix);
                Matrix::activeBackward(_activeFunctionType, AMatrix, dAMatrix, XMatrix, dXMatrix);
            }
            break;
        case cf_LogLikelihood:
            if (_activeFunctionType == af_Softmax)
            {
                Matrix::add(YMatrix, -1, AMatrix, dAMatrix);
                Matrix::cpyData(dXMatrix, dAMatrix);
            }
            break;
        default:
            break;
        }
    }
    else
    {
        activeBackward2();
    }
}


