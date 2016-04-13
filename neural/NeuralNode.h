#pragma once
#include <vector>
#include <map>
#include <functional>

#include "ActiveFunctions.h"

//��������
class NeuralBond
{
public:
	double weight = 0;
	double learnSpeed = 1;
	class NeuralNode* startNode;
	class NeuralNode* endNode;

	void updateWeight(double learnSpeed);
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
	std::string tag;

	std::map<NeuralNode*, NeuralBond> prevBonds;  //�������ֻ����weight������
	std::map<NeuralNode*, NeuralBond> nextBonds;  //�������ֻ����weight������

	double outputValue;
	double totalInputValue;
	void collectInputValue();
	void activeOutputValue();

	ActiveFunctions af;

	//feedback��active�ĵ���
	std::function<double(double)> activeFunction = af.linear;
	std::function<double(double)> feedbackFunction = af.dlinear;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _feedback);
	void connect(NeuralNode* node, double w = 0);

	void setWeight(NeuralNode* node, double w = 0);

	void updateWeight(NeuralNode* startNode, NeuralNode* endNode, double learnSpeed, double delta);

	double delta;
	void updateDelta(double expect = 0);

};

