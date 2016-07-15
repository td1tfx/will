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

//ѧϰģʽ
typedef enum
{
    nl_Whole = 0,          //ȫ������һ��ѧϰ�����ݶ��ʱ�����������
    nl_Online = 1,         //ÿ��ѧϰһ������ʵ����������Ĵ���
    nl_MiniBatch = 2,      //ÿ��ѧϰһС�����������һ����Ӱ����ܻ�Ƚϴ�
} NetBatchType;


//����
class Net : Neural
{
public:
    Net();
    virtual ~Net();

    int Id;

    int MaxGroup = 100000;  //һ���ܴ���������������ڴ���Դ��С���

    int InputNodeCount;
    int OutputNodeCount;

    //ѵ����
    Matrix* trainX = nullptr;
    Matrix* trainY = nullptr;
    int train_groupCount = 0;
    //���Լ�
    Matrix* testX = nullptr;
    Matrix* testY = nullptr;
    int test_groupCount = 0;

    void run(Option* option);

    //�񾭲�
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

    real LearnSpeed = 0.5;  //ѧϰ�ٶ�
    void setLearnSpeed(real s) { LearnSpeed = s; }

    real Lambda = 0.0;      //���򻯲�������ֹ�����
    void setRegular(real l) { Lambda = l; }

    ActiveFunctionType WorkType = af_Sigmoid;  //ʵ�ʾ������һ��ļ���
    void setWorkType(ActiveFunctionType wt);

    void createLayers(int layerCount);  //��������������

    void train(int times = 1000000, int interval = 1000, real tol = 1e-3, real dtol = 0);  //ѵ������

    void active(Matrix* X, Matrix* Y, Matrix* A, int groupCount, int batchCount,
        bool learn = false, real* error = nullptr);  //����һ�����

    //void setInputData(d_matrix* input, int groupid);
    //void setExpectData(d_matrix* expect, int groupid);

    void getOutputData(Matrix* M, int groupCount, int col = 0);

    void readData(const char* filename, int* count, Matrix** pX, Matrix** pY);
    int resetGroupCount(int n);

    //��������
    virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //������������д����
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
