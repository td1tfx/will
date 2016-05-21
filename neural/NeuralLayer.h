#pragma once
#include <vector>
#include <functional>
#include <string>
#include "ActiveFunctions.h"

typedef enum 
{
	HaveNotConstNode = 0,
	HaveConstNode = 1,

}NeuralLayerMode;

//��Ԫ�����ͣ��������أ����룬���
typedef enum
{
	Hidden,
	Input,
	Output,
	Const,
} NeuralLayerType;

//�񾭲�
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int id;

	int nodeAmount;
	int groupAmount;
	double* data = nullptr;
	double* weight = nullptr;

	NeuralLayer* prevLayer;

	void initData(int nodeAmount, int groupAmount);

	double& getValue(int nodeid, int groupid) { return data[groupid*nodeAmount+nodeid]; }
	void setValue(double value, int nodeid, int groupid) { data[groupid*nodeAmount + nodeid] = value; }
	static void connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
	void markMax(int groupid);
	void normalized();

	//feedback��active�ĵ���
	std::function<double(double)> activeFunction = ActiveFunctions::sigmoid;
	std::function<double(double)> dactiveFunction = ActiveFunctions::dsigmoid;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive);
};

