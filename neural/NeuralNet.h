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

	void learn(void* input, void* output);

	void train(void* input, void* output);

	//��������
	void setLayers(); //������������д����

	//����
	void* inputData = nullptr, *outputData = nullptr;
	void readData(std::string& filename);

};

