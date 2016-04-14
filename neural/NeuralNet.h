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

	double learnSpeed = 0.1;

	void createLayers(int layerAmount);  //��������������

	void learn(double* input, double* output);  //ѧϰһ������

	void train(int times = 1000000, double tol = 0.0001);  //ѧϰһ������
	void test();  

	void activeOutputValue(double* input, double* output, int amount = -1);  //����һ�����

	//����
	double* inputData = nullptr;
	double* outputData = nullptr;
	void readData(std::string& filename);


	//��������
	virtual void setLayers(); //������������д����
	void outputWeight(); //������������д����

};

