#pragma once
#include <vector>
#include <functional>
#include <string>
#include "ActiveFunctions.h"
#include "MatrixFunctions.h"

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
	//Const,
} NeuralLayerType;

//�񾭲�
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int id;

	int nodeAmount;
	static int groupAmount;
	
	double* data = nullptr;
	double* expect = nullptr;
	double* delta = nullptr;
	//data��ʽ�������ǽڵ�������������������
	
	double* weight = nullptr;
	//weight��ʽ�������Ǳ���Ľڵ�������������һ��Ľڵ���

	NeuralLayer* prevLayer;
	NeuralLayer* nextLayer;

	void initData(int nodeAmount, int groupAmount);
	double& getData(int nodeid, int groupid) { return data[groupid + nodeid*groupAmount]; }
	
	void initExpect();
	double& getExpect(int nodeid, int groupid) { return expect[groupid + nodeid*groupAmount]; }

	void setData(double value, int nodeid, int groupid) { data[groupid + nodeid*groupAmount] = value; }
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

	void activeOutputValue();

	void updateDelta();
	void backPropagate(double learnSpeed = 0.5);

	NeuralLayerMode mode = HaveConstNode;
	NeuralLayerType type = Hidden;

};

