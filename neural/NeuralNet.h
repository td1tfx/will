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

//ѧϰģʽ
typedef enum 
{
	nl_Whole = 0,          //ȫ������һ��ѧϰ�����ݶ��ʱ�����������
	nl_Online = 1,         //ÿ��ѧϰһ������ʵ����������Ĵ���
	nl_MiniBatch = 2,      //ÿ��ѧϰһС�����������һ����Ӱ����ܻ�Ƚϴ�
} NeuralNetLearnType;


//����ģʽ
typedef enum
{
	nw_Fit = 0,            //���
	nw_Classify = 1,       //���࣬��ɸѡ���ֵ��Ϊ1��������Ϊ0
	nw_Probability = 2,    //���ʣ�������һ��	
} NeuralNetWorkType;


//����
class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();
	int Id;

	Option _option;
	void loadOptoin(const char* filename);
	int MaxGroup = 100000;  //һ���ܴ���������������ڴ���Դ��С���

	void run();

	//�񾭲�
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
	void setLearnMode(NeuralNetLearnType lm, int lb = -1);

	double LearnSpeed = 0.5;  //ѧϰ�ٶ�
	void setLearnSpeed(double s) { LearnSpeed = s; }

	double Lambda = 0.0;      //���򻯲�������ֹ�����
	void setRegular(double l) { Lambda = l; }

	NeuralNetWorkType WorkMode = nw_Fit;
	void setWorkMode(NeuralNetWorkType wm);

	void createLayers(int layerCount);  //��������������

	void train(int times = 1000000, int interval = 1000, double tol = 1e-3, double dtol = 0);  //ѵ������
	
	void active(d_matrix* input, d_matrix* expect, d_matrix* output, int groupCount, int batchCount,
		bool learn = false, double* error = nullptr);  //����һ�����

	//void setInputData(d_matrix* input, int groupid);
	//void setExpectData(d_matrix* expect, int groupid);

	void getOutputData(d_matrix* output, int groupCount, int col=0);

	//����
	d_matrix* _train_inputData = nullptr;
	d_matrix* _train_expectData = nullptr;
	int _train_groupCount = 0;   //ʵ�ʵ�������
	
	typedef enum { Train, Test } DateMode;
	void readData(const char* filename, DateMode dm = Train);
	int resetGroupCount(int n);

	d_matrix* _test_inputData = nullptr;
	d_matrix* _test_expectData = nullptr;
	int _test_groupCount = 0;

	//��������
	virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //������������д����
	void saveInfo(const char* filename = nullptr); 
	void createByLoad(const char* filename);

	//NeuralNetCalMode activeMode = ByNode;
	//NeuralNetCalMode backPropageteMode = ByNode;

	void readMNIST();

	void selectTest();
	void test();
	void printResult(int nodeCount, int groupCount, d_matrix* output, d_matrix* expect);
};

