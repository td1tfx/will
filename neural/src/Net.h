#pragma once
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include "libconvert.h"
#include "Option.h"
#include "Layer.h"
#include "LayerConvolution.h"
#include "LayerFull.h"
#include "LayerPooling.h"
#include "Test.h"
#include "Neural.h"

//学习模式
typedef enum
{
    nl_Whole = 0,          //全部数据一起学习，数据多的时候收敛会很慢
    nl_Online = 1,         //每次学习一个，其实可以用下面的代替
    nl_MiniBatch = 2,      //每次学习一小批，但是最后一批的影响可能会比较大
} NetBatchType;


//神经网
class Net : Neural
{
public:
    Net();
    virtual ~Net();

    int Id;

    int MaxGroup = 100000;  //一次能处理的数据量，与内存或显存大小相关

    int InputNodeCount;
    int OutputNodeCount;

    //训练集
    Matrix* trainX = nullptr;
    Matrix* trainY = nullptr;
    int train_groupCount = 0;
    //测试集
    Matrix* testX = nullptr;
    Matrix* testY = nullptr;
    int test_groupCount = 0;

    void run(Option* option);

    //神经层
    Layer** Layers;
    int LayerCount = 0;
    //std::vector<NeuralLayer*>& getLayerVector() { return Layers; }
    Layer*& getLayer(int number) { return Layers[number]; }
    Layer*& getFirstLayer() { return Layers[0]; }
    Layer*& getLastLayer() { return Layers[LayerCount - 1]; }
    int getLayerCount() { return LayerCount; };

    Layer* createLayer(LayerConnectionType mode);

    NetBatchType BatchType = nl_Whole;
    int MiniBatchCount = -1;
    void setBatchType(NetBatchType lm, int lb = -1);

    real LearnSpeed = 0.5;  //学习速度
    void setLearnSpeed(real s) { LearnSpeed = s; }

    real Lambda = 0.0;      //正则化参数，防止过拟合
    void setRegular(real l) { Lambda = l; }

    ActiveFunctionType WorkType = af_Sigmoid;  //实际就是最后一层的激活
    void setWorkType(ActiveFunctionType wt);

    void createLayers(int layerCount);  //包含输入和输出层

    void train(int times = 1000000, int interval = 1000, real tol = 1e-3, real dtol = 0);  //训练过程

    void active(Matrix* X, Matrix* Y, Matrix* A, int groupCount, int batchCount,
        bool learn = false, real* error = nullptr);  //计算一组输出

    //void setInputData(d_matrix* input, int groupid);
    //void setExpectData(d_matrix* expect, int groupid);

    void getOutputData(Matrix* M, int groupCount, int col = 0);

    void readData(const char* filename, int* count, Matrix** pX, Matrix** pY);
    int resetGroupCount(int n);

    //具体设置
    virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //具体的网络均改写这里
    void saveInfo(const char* filename = nullptr);
    void createByLoad(const char* filename);

    //NeuralNetCalMode activeMode = ByNode;
    //NeuralNetCalMode backPropageteMode = ByNode;

    void readMNIST();

    void selectTest();
    void test(int forceOutput = 0, int testMax = 0);
    void extraTest(const char* filename, int forceOutput = 0, int testMax = 0);
    void outputTest(const char* info, int nodeCount, int groupCount, Matrix* X, Matrix* Y, int forceOutput, int testMax);
};
