#pragma once
#include <vector>
#include <functional>
#include <string>
#include "ActiveFunctions.h"
#include "MatrixFunctions.h"


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
	static int step;

	NeuralLayerType type = Hidden;
	
	//data��ʽ�������ǽڵ�������������������
	d_matrix* input = nullptr;
	d_matrix* output = nullptr;
	d_matrix* expect = nullptr;
	d_matrix* delta = nullptr;
	d_matrix* bias = nullptr;
	d_matrix* bias_as = nullptr;

	void deleteData();
	
	//weight��ʽ�������Ǳ���Ľڵ�������������һ��Ľڵ���
	d_matrix* weight = nullptr;	

	NeuralLayer* prevLayer;
	NeuralLayer* nextLayer;

	void initData(int nodeAmount, int groupAmount, NeuralLayerType type = Hidden);
	void resetData(int groupAmount);
	double& getOutput(int nodeid, int groupid) { return output->getData(nodeid, groupid); }
	
	double& getExpect(int nodeid, int groupid) { return expect->getData(nodeid, groupid); }

	static void connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
	void markMax();
	void normalized();

	//dactive��active�ĵ���
	std::function<double(double)> activeFunction = ActiveFunctions::sigmoid;
	std::function<double(double)> dactiveFunction = ActiveFunctions::dsigmoid;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive);

	void activeOutputValue();

	void updateDelta();
	void backPropagate(double learnSpeed = 0.5, double lambda = 0.1);

};

