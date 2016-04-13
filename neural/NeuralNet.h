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

	int inputNodeAmount;
	int outputNodeAmount;
	int dataGroupAmount;

	double learnSpeed = 1;

	void createLayers(int layerAmount);  //包含输入和输出层

	void learn(double* input, double* output);  //学习一组数据

	void train();  //学习一批数据

	void activeOutputValue(double* input, double* output);  //计算一组输出

	//数据
	double* inputData = nullptr;
	double* outputData = nullptr;
	void readData(std::string& filename);


	//具体设置
	void setLayers(); //具体的网络均改写这里

};

