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

	int inputNodeAmount;
	int outputNodeAmount;
	int dataGroupAmount;

	double learnSpeed = 1;

	void createLayers(int layerAmount);  //��������������

	void learn(double* input, double* output);  //ѧϰһ������

	void train();  //ѧϰһ������

	void activeOutputValue(double* input, double* output);  //����һ�����

	//����
	double* inputData = nullptr;
	double* outputData = nullptr;
	void readData(std::string& filename);


	//��������
	void setLayers(); //������������д����

};

