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

	//神经层
	std::vector<NeuralLayer*> layers;

	NeuralLayer*& getLayer(int number) { return (layers.at(number)); }
	int getLayerAmount() { return layers.size(); };

	int inputAmount;
	int outputAmount;
	int dataAmount;

	double learnSpeed = 0.1;

	void createLayers(int layerAmount);  //包含输入和输出层

	void learn(double* input, double* output);  //学习一组数据

	void train(int times = 1000000, double tol = 0.0001);  //学习一批数据
	
	void activeOutputValue(double* input, double* output, int amount = -1);  //计算一组输出

	//数据
	double* inputData = nullptr;
	double* expectData = nullptr;
	void readData(std::string& filename, double* input =nullptr, double* output = nullptr, int amount = -1);

	std::vector<bool> isTest;
	double* inputTestData = nullptr;
	double* expectTestData = nullptr;
	int testDataAmount;
	void selectTest();
	void test();

	//具体设置
	virtual void setLayers(double learnSpeed = 0.5, int layerAmount = 3, bool haveConstNode = true); //具体的网络均改写这里
	void outputWeight(); //具体的网络均改写这里
	void createByFile(std::string& filename);

};

