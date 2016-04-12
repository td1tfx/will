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

	int inputNodeAmount;
	int outputNodeAmount;
	int dataGroupAmount;

	void createLayers(int layerAmount);  //包含输入和输出层

	void learn(double* input, double* output);

	void train(double* input, double* output);

	void calOutput(double* input, double* output);

	//具体设置
	void setLayers(); //具体的网络均改写这里

	//数据
	double* inputData = nullptr;
	double* outputData = nullptr;
	void readData(std::string& filename);

};

