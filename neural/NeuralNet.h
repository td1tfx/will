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
	std::vector<NeuralLayer*> layers;
	std::vector<NeuralLayer*>& getLayerVector() { return layers; }

	int id;

	NeuralLayer*& getLayer(int number) { return layers[number]; }
	NeuralLayer*& getFirstLayer() { return layers[0]; }
	NeuralLayer*& getLastLayer() { return layers.back(); }
	int getLayerAmount() { return layers.size(); };

	int inputAmount;
	int outputAmount;
	int trainDataAmount = 0;  //ѵ����������
	int realDataAmount = 0;   //ʵ�ʵ�������
	int nodeDataAmount = 0;   //�ڵ��������

	NeuralNetLearnMode learnMode = Batch;

	double learnSpeed = 0.5;  //ѧϰ�ٶ�
	void setLearnSpeed(double s) { learnSpeed = s; }

	double lambda = 0.0;      //���򻯲�������ֹ�����
	void setRegular(double l) { lambda = l; }

	void setLearnMode(NeuralNetLearnMode lm);

	NeuralNetWorkMode workMode = Fit;
	void setWorkMode(NeuralNetWorkMode wm);

	void createLayers(int amount);  //��������������

	void learn();

	void train(int times = 1000000, int interval = 1000, double tol = 1e-3, double dtol = 1e-9);  //ѵ������
	
	void activeOutputValue(double* input, double* output, int amount);  //����һ�����

	void setInputData(double* input, int nodeAmount, int groupAmount);
	void getOutputData(double* output, int nodeAmount, int groupAmount);
	void setExpectData(double* expect, int nodeAmount, int groupAmount);

	//����
	double* inputData = nullptr;
	double* expectData = nullptr;
	void readData(const char* filename);

	std::vector<bool> isTest;
	double* inputTestData = nullptr;
	double* expectTestData = nullptr;
	int testDataAmount = 0;
	void selectTest();
	void test();

	//��������
	virtual void createByData(int layerAmount = 3, int nodesPerLayer = 7); //������������д����
	void outputBondWeight(const char* filename = nullptr); 
	void createByLoad(const char* filename);

	//NeuralNetCalMode activeMode = ByNode;
	//NeuralNetCalMode backPropageteMode = ByNode;

	void readMNIST();

};

