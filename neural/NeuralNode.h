#pragma once
#include <vector>
#include <map>
#include <functional>

#include "ActiveFunctions.h"

//键连接类
class NeuralBond
{
public:
	double weight = 0;
	double learnSpeed = 1;
	class NeuralNode* startNode;
	class NeuralNode* endNode;

	void updateWeight(double learnSpeed);
};

//类型，隐藏，输入，输出
typedef enum NeuralNodeType {
	Hidden,
	Input,
	Output,
} NeuralNodeType;

//节点
class NeuralNode
{
public:
	NeuralNode();
	virtual ~NeuralNode();

	NeuralNodeType type;
	std::string tag;

	std::map<NeuralNode*, NeuralBond*> prevBonds;  //这里好像只保存weight就行了
	std::map<NeuralNode*, NeuralBond*> nextBonds;  //next实际为prev的镜像，start和end相同时保存的是同一个指针

	NeuralBond*& getPrevBond(NeuralNode* node) { return prevBonds[node]; };
	NeuralBond*& getNextBond(NeuralNode* node) { return nextBonds[node]; };  

	double outputValue;
	double inputValue;
	double expect;

	void setExpect(double expect, int i = -1);

	void collectInputValue();
	void activeOutputValue();

	ActiveFunctions af;

	//feedback是active的导数
	std::function<double(double)> activeFunction = af.linear;
	std::function<double(double)> feedbackFunction = af.dlinear;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _feedback);
	void connect(NeuralNode* node, double w = 0);

	void setWeight(NeuralNode* node, double w = 0);

	void updateWeight(NeuralNode* startNode, NeuralNode* endNode, double learnSpeed, double delta);

	double delta;
	void updateOneDelta();

	//多组数据
	int dataGroupAmount = 0;
	std::vector<double> outputValues;
	std::vector<double> inputValues;
	std::vector<double> expects;
	std::vector<double> deltas;

	void setDataGroupAmount(int n);
	void updateDelta();

};

