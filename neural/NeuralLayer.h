#pragma once
#include <vector>
#include <functional>
#include <string>
#include "ActiveFunctions.h"
#include "MatrixFunctions.h"


//神经元的类型，包含隐藏，输入，输出
typedef enum
{
	Hidden,
	Input,
	Output,
	//Const,
} NeuralLayerType;

//神经层
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
	
	//data格式：行数是节点数，列数是数据组数
	d_matrix* input = nullptr;
	d_matrix* output = nullptr;
	d_matrix* expect = nullptr;
	d_matrix* delta = nullptr;
	d_matrix* bias = nullptr;
	d_matrix* bias_as = nullptr;

	void deleteData();
	
	//weight格式：行数是本层的节点数，列数是上一层的节点数
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

	//dactive是active的导数
	std::function<double(double)> activeFunction = ActiveFunctions::sigmoid;
	std::function<double(double)> dactiveFunction = ActiveFunctions::dsigmoid;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive);

	void activeOutputValue();

	void updateDelta();
	void backPropagate(double learnSpeed = 0.5, double lambda = 0.1);

};

