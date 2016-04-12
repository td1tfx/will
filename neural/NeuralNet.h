#pragma once
#include <stdio.h>
#include <vector>
#include "NeuralLayer.h"
#include "NeuralNode.h"

class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();

	std::vector<NeuralLayer*> layers;

	NeuralLayer*& getLayer(int number) { return (layers.at(number)); }

	int inputNodeNumber;
	int outputNodeNumber;

	void createLayers(int layerAmount);  //包含输入和输出层

	void learn(void* input, void* output);

	void train(void* input, void* output);

};

