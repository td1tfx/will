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

	NeuralLayer*& getLayer(int number) { return (layers.at(number)); }
	int getLayerAmount() { return layers.size(); };

	int inputAmount;
	int outputAmount;
	int dataAmount;

	double learnSpeed = 0.1;

	void createLayers(int layerAmount);  //��������������

	void learn(double* input, double* output);  //ѧϰһ������

	void train(int times = 1000000, double tol = 0.0001);  //ѧϰһ������
	
	void activeOutputValue(double* input, double* output, int amount = -1);  //����һ�����

	//����
	double* inputData = nullptr;
	double* expectData = nullptr;
	void readData(std::string& filename, double* input =nullptr, double* output = nullptr, int amount = -1);

	std::vector<bool> isTest;
	double* inputTestData = nullptr;
	double* expectTestData = nullptr;
	int testDataAmount;
	void selectTest();
	void test();

	//��������
	virtual void setLayers(double learnSpeed = 0.5, int layerAmount = 3, bool haveConstNode = true); //������������д����
	void outputWeight(); //������������д����
	void createByFile(std::string& filename);

};

