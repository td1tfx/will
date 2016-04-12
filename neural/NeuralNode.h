#pragma once
#include <vector>
#include <map>
#include <functional>

#include "ActiveFunctions.h"

//键连接类
class NeuralBond
{
public:
	double weight;
	class NeuralNode* startNode;
	class NeuralNode* endNode;
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

	std::map<NeuralNode*, NeuralBond> bonds;  //这里好像只保存weight就行了

	double outputValue;
	double totalInputValue;
	void collectInputValue();
	void activeOutputValue();

	std::function<double(double)> activeFunction = ActiveFunctions::linear;
	std::function<double(double)> feedbackFunction = ActiveFunctions::linear;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _feedback);
	void connect(NeuralNode* node, double w = 0);

	void setWeight(NeuralNode* node, double w = 0);

};

