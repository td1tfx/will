#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include <cmath>
#include "NeuralLayer.h"
#include "MNISTFunctions.h"
#include "Option.h"
#include "NeuralLayerFactory.h"
#include "lib/libconvert.h"

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

	Option _option;
	void loadOption(const char* filename);
	int MaxGroup = 100000;  //一次能处理的数据量，与内存或显存大小相关

	void run();

	//神经层
	NeuralLayer** Layers;
	int LayerCount = 0;
	//std::vector<NeuralLayer*>& getLayerVector() { return Layers; }
	NeuralLayer*& getLayer(int number) { return Layers[number]; }
	NeuralLayer*& getFirstLayer() { return Layers[0]; }
	NeuralLayer*& getLastLayer() { return Layers[LayerCount - 1]; }
	int getLayerCount() { return LayerCount; };

	int InputNodeCount;
	int OutputNodeCount;

	NeuralNetLearnType BatchMode = nl_Whole;
	int MiniBatchCount = -1;
	void setLearnType(NeuralNetLearnType lm, int lb = -1);

	double LearnSpeed = 0.5;  //学习速度
	void setLearnSpeed(double s) { LearnSpeed = s; }

	double Lambda = 0.0;      //正则化参数，防止过拟合
	void setRegular(double l) { Lambda = l; }

	NeuralNetWorkType WorkType = nw_Fit;
	void setWorkType(NeuralNetWorkType wm);

	void createLayers(int layerCount);  //包含输入和输出层

	void train(int times = 1000000, int interval = 1000, double tol = 1e-3, double dtol = 0);  //训练过程

	void active(Matrix* input, Matrix* expect, Matrix* output, int groupCount, int batchCount,
		bool learn = false, double* error = nullptr);  //计算一组输出

	//void setInputData(d_matrix* input, int groupid);
	//void setExpectData(d_matrix* expect, int groupid);

	void getOutputData(Matrix* output, int groupCount, int col = 0);

	//数据
	Matrix* _train_inputData = nullptr;
	Matrix* _train_expectData = nullptr;
	int _train_groupCount = 0;   //实际的数据量

	void readData(const char* filename, int* count, Matrix** input, Matrix** expect);
	int resetGroupCount(int n);

	Matrix* _test_inputData = nullptr;
	Matrix* _test_expectData = nullptr;
	int _test_groupCount = 0;

	//具体设置
	virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //具体的网络均改写这里
	void saveInfo(const char* filename = nullptr);
	void createByLoad(const char* filename);

	//NeuralNetCalMode activeMode = ByNode;
	//NeuralNetCalMode backPropageteMode = ByNode;

	void readMNIST();

	void selectTest();
	void test();
	void printResult(int nodeCount, int groupCount, Matrix* output, Matrix* expect);
};

