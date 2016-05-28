#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include "NeuralLayer.h"
#include "NeuralNode.h"
#include "lib/libconvert.h"


//ѧϰģʽ
typedef enum 
{
	Online,
	Batch,
	//�����������0����϶࣬����ѧϰ��ȽϿ�
	//ͨ�����������ѧϰ�ῼ��ȫ�����ȣ�ӦΪ��ѡ
	//����ѧϰÿ�ζ��������м���ֵ������ѧϰÿһ�����ݸ���һ�μ���ֵ
} NeuralNetLearnMode;

//����ģʽ
typedef enum 
{
	ByLayer,
	ByNode,
} NeuralNetCalMode;

//����ģʽ
typedef enum
{
	Fit,  //���
	Classify,  //���࣬��ɸѡ���ֵ��Ϊ1��������Ϊ0
	Probability,   //���ʣ�������һ��	
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

	std::vector<NeuralNode*> nodes;
	void initNodes();

	int id;

	NeuralLayer*& getLayer(int number) { return layers[number]; }
	NeuralLayer*& getFirstLayer() { return layers[0]; }
	NeuralLayer*& getLastLayer() { return layers[layers.size() - 1]; }
	int getLayerAmount() { return layers.size(); };

	int inputAmount;
	int outputAmount;
	int realDataAmount = 0;  //ʵ�ʵ�������
	int nodeDataAmount = 0;  //�ڵ��������

	NeuralNetLearnMode learnMode = Batch;

	double learnSpeed = 0.5;
	void setLearnSpeed(double s) { learnSpeed = s; }
	void setLearnMode(NeuralNetLearnMode lm);

	NeuralNetWorkMode workMode = Fit;
	void setWorkMode(NeuralNetWorkMode wm) { workMode = wm; }

	void createLayers(int amount);  //��������������

	void learn(double* input, double* output, int amount);  //ѧϰһ������

	void train(int times = 1000000, double tol = 0.01);  //ѧϰһ������
	
	double calTol();

	void activeOutputValue(double* input, double* output, int amount);  //����һ�����

	//����
	double* inputData = nullptr;
	double* expectData = nullptr;
	void readData(const char* filename, double* input = nullptr, double* output = nullptr, int amount = -1);

	std::vector<bool> isTest;
	double* inputTestData = nullptr;
	double* expectTestData = nullptr;
	int testDataAmount = 0;
	void selectTest();
	void test();

	//��������
	virtual void createByData(NeuralLayerMode layerMode = HaveConstNode, int layerAmount = 3, int nodesPerLayer = 7); //������������д����
	void outputBondWeight(const char* filename = nullptr); //������������д����
	void createByLoad(const char* filename, bool haveConstNode = true);

	void setNodeDataAmount(int amount);

	NeuralNetCalMode activeMode = ByLayer;
	NeuralNetCalMode backPropageteMode = ByLayer;

};

