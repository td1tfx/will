#pragma once
#include <stdio.h>
#include <vector>
#include "NeuralLayer.h"
#include "NeuralNode.h"
#include "libconvert.h"

class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();

	//�񾭲�
	std::vector<NeuralLayer*> layers;
	std::vector<NeuralLayer*>& getLayers() { return layers; }

	int id;

	NeuralLayer*& getLayer(int number) { return layers[number]; }
	NeuralLayer*& getFirstLayer() { return layers[0]; }
	NeuralLayer*& getLastLayer() { return layers[layers.size() - 1]; }
	int getLayerAmount() { return layers.size(); };

	int inputAmount;
	int outputAmount;
	int dataAmount = 0;

	double learnSpeed = 0.5;
	void setLearnSpeed(double s) { learnSpeed = s; }

	void createLayers(int amount);  //��������������

	void learn(double* input, double* output);  //ѧϰһ������

	void train(int times = 1000000, double tol = 0.01);  //ѧϰһ������
	
	void activeOutputValue(double* input, double* output, int amount = -1);  //����һ�����

	//����
	double* inputData = nullptr;
	double* expectData = nullptr;
	void readData(const std::string& filename, double* input =nullptr, double* output = nullptr, int amount = -1);

	std::vector<bool> isTest;
	double* inputTestData = nullptr;
	double* expectTestData = nullptr;
	int testDataAmount = 0;
	void selectTest();
	void test();

	//��������
	virtual void createByData(bool haveConstNode = true, int layerAmount = 3); //������������д����
	void outputWeight(); //������������д����
	void createByLoad(const std::string& filename, bool haveConstNode = true);

};

