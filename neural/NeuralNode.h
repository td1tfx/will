#pragma once
#include <vector>
#include <map>
#include <functional>

#include "ActiveFunctions.h"

//��������
class NeuralBond
{
public:
	double weight;
	class NeuralNode* startNode;
	class NeuralNode* endNode;
};

//���ͣ����أ����룬���
typedef enum NeuralNodeType {
	Hidden,
	Input,
	Output,
} NeuralNodeType;

//�ڵ�
class NeuralNode
{
public:
	NeuralNode();
	virtual ~NeuralNode();

	NeuralNodeType type;

	std::map<NeuralNode*, NeuralBond> bonds;  //�������ֻ����weight������

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

