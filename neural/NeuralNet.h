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
	Batch = 0,
	Online = 1,
	MiniBatch = 2,
	//�����������0����϶࣬����ѧϰ��ȽϿ�
	//ͨ�����������ѧϰ�ῼ��ȫ�����ȣ�ӦΪ��ѡ
	//����ѧϰÿ�ζ��������м���ֵ������ѧϰÿһ�����ݸ���һ�μ���ֵ
} NeuralNetLearnMode;

//����ģʽ��no use��
/*
typedef enum 
{
	ByLayer,
	ByNode,
} NeuralNetCalMode;
*/

//����ģʽ
typedef enum
{
	Fit = 0,            //���
	Classify = 1,       //���࣬��ɸѡ���ֵ��Ϊ1��������Ϊ0
	Probability = 2,    //���ʣ�������һ��	
} NeuralNetWorkMode;


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
	std::vector<NeuralLayer*> Layers;
	std::vector<NeuralLayer*>& getLayerVector() { return Layers; }
	NeuralLayer*& getLayer(int number) { return Layers[number]; }
	NeuralLayer*& getFirstLayer() { return Layers[0]; }
	NeuralLayer*& getLastLayer() { return Layers.back(); }
	int getLayerCount() { return Layers.size(); };

	int InputNodeCount;
	int OutputNodeCount;

	NeuralNetLearnMode LearnMode = Batch;
	int MiniBatchCount = -1;
	void setLearnMode(NeuralNetLearnMode lm, int lb = -1);

	double LearnSpeed = 0.5;  //ѧϰ�ٶ�
	void setLearnSpeed(double s) { LearnSpeed = s; }

	double Lambda = 0.0;      //���򻯲�������ֹ�����
	void setRegular(double l) { Lambda = l; }

	NeuralNetWorkMode WorkMode = Fit;
	void setWorkMode(NeuralNetWorkMode wm);

	void createLayers(int layerCount);  //��������������

	void train(int times = 1000000, int interval = 1000, double tol = 1e-3, double dtol = 0);  //ѵ������
	
	void active(d_matrix* input, d_matrix* expect, d_matrix* output, int groupCount, int batchCount,
		bool learn = false, double* error = nullptr);  //����һ�����

	void setInputData(d_matrix* input, int groupid);
	void setExpectData(d_matrix* expect, int groupid);

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

