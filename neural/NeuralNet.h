#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include "NeuralLayer.h"
#include "lib/libconvert.h"
#include "MNISTFunctions.h"


//ѧϰģʽ
typedef enum 
{
	Online,
	Batch,
	MiniBatch,
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
	Fit,            //���
	Classify,       //���࣬��ɸѡ���ֵ��Ϊ1��������Ϊ0
	Probability,    //���ʣ�������һ��	
} NeuralNetWorkMode;


//����
class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();

	//�񾭲�
	std::vector<NeuralLayer*> Layers;
	std::vector<NeuralLayer*>& getLayerVector() { return Layers; }

	int Id;

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

	void learn();

	void train(int times = 1000000, int interval = 1000, double tol = 1e-3, double dtol = 1e-9);  //ѵ������
	
	void activeOutputValue(double* input, double* output, int groupCount);  //����һ�����

	void setInputData(double* input, int nodeCount, int groupid);
	void setExpectData(double* expect, int nodeCount, int groupid);

	void getOutputData(int nodeCount, int groupCount, double* output);

	//����
	double* _train_inputData = nullptr;
	double* _train_expectData = nullptr;
	int _train_groupCount = 0;   //ʵ�ʵ�������
	void readData(const char* filename);
	void resetGroupCount(int n);

	double* _test_inputData = nullptr;
	double* _test_expectData = nullptr;
	int _test_groupCount = 0;
	void selectTest();
	void test();
	void printResult(int nodeCount, int groupCount, double* output, double* expect);

	//��������
	virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //������������д����
	void outputBondWeight(const char* filename = nullptr); 
	void createByLoad(const char* filename);

	//NeuralNetCalMode activeMode = ByNode;
	//NeuralNetCalMode backPropageteMode = ByNode;

	void readMNIST();

};

