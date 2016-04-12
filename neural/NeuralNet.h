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

	void learn(void* input, void* output);

	void train(void* input, void* output);

	//具体设置
	void setLayers(); //具体的网络均改写这里

	//数据
	void* inputData = nullptr, *outputData = nullptr;
	void readData(std::string& filename);

};

