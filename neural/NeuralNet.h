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

	int inputNodeAmount;
	int outputNodeAmount;
	int dataGroupAmount;

	void createLayers(int layerAmount);  //��������������

	void learn(double* input, double* output);

	void train(double* input, double* output);

	void calOutput(double* input, double* output);

	//��������
	void setLayers(); //������������д����

	//����
	double* inputData = nullptr;
	double* outputData = nullptr;
	void readData(std::string& filename);

};

