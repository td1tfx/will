#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include <cmath>
#include "lib/libconvert.h"
#include "Option.h"
#include "NeuralLayerFactory.h"
#include "Test.h"


//学习模式
typedef enum
{
	nl_Whole = 0,          //全部数据一起学习，数据多的时候收敛会很慢
	nl_Online = 1,         //每次学习一个，其实可以用下面的代替
	nl_MiniBatch = 2,      //每次学习一小批，但是最后一批的影响可能会比较大
} NeuralNetLearnType;


//工作模式
typedef enum
{
	nw_Fit = 0,            //拟合
	nw_Classify = 1,       //分类，会筛选最大值设为1，其他设为0
	nw_Probability = 2,    //几率，结果会归一化	
} NeuralNetWorkType;


//神经网
class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();
	int Id;

	int MaxGroup = 100000;  //一次能处理的数据量，与内存或显存大小相关

	int InputNodeCount;
	int OutputNodeCount;

	//训练集
	Matrix* train_input = nullptr;
	Matrix* train_expect = nullptr;
	int train_groupCount = 0;
	//测试集
	Matrix* test_input = nullptr;
	Matrix* test_expect = nullptr;
	int test_groupCount = 0;

	void run(Option* option);

	//神经层
	NeuralLayer** Layers;
	int LayerCount = 0;
	//std::vector<NeuralLayer*>& getLayerVector() { return Layers; }
	NeuralLayer*& getLayer(int number) { return Layers[number]; }
	NeuralLayer*& getFirstLayer() { return Layers[0]; }
	NeuralLayer*& getLastLayer() { return Layers[LayerCount - 1]; }
	int getLayerCount() { return LayerCount; };

	NeuralNetLearnType BatchMode = nl_Whole;
	int MiniBatchCount = -1;
	void setLearnType(NeuralNetLearnType lm, int lb = -1);

	real LearnSpeed = 0.5;  //学习速度
	void setLearnSpeed(real s) { LearnSpeed = s; }

	real Lambda = 0.0;      //正则化参数，防止过拟合
	void setRegular(real l) { Lambda = l; }

	NeuralNetWorkType WorkType = nw_Fit;
	void setWorkType(NeuralNetWorkType wm);

	void createLayers(int layerCount);  //包含输入和输出层

	void train(int times = 1000000, int interval = 1000, real tol = 1e-3, real dtol = 0);  //训练过程

	void active(Matrix* input, Matrix* expect, Matrix* output, int groupCount, int batchCount,
		bool learn = false, real* error = nullptr);  //计算一组输出

	//void setInputData(d_matrix* input, int groupid);
	//void setExpectData(d_matrix* expect, int groupid);

	void getOutputData(Matrix* output, int groupCount, int col = 0);

	void readData(const char* filename, int* count, Matrix** input, Matrix** expect);
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
	void outputTest(const char* info, int nodeCount, int groupCount, Matrix* input, Matrix* expect, int forceOutput, int testMax);
};

