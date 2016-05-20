#pragma once
#include <vector>
#include <map>
#include <functional>
#include <string>
#include "ActiveFunctions.h"

//��
class NeuralBond
{
public:
	double weight = 0;
	double learnSpeed = 1;
	class NeuralNode* startNode;
	class NeuralNode* endNode;

	void updateWeight(double learnSpeed);
};


//��Ԫ�����ͣ��������أ����룬���
typedef enum 
{
	Hidden,
	Input,
	Output,
	Const,
} NeuralNodeType;

//��Ԫ���񾭽ڵ�
class NeuralNode
{
public:
	NeuralNode();
	virtual ~NeuralNode();

	NeuralNodeType type;
	std::string tag;
	int id;

	std::map<NeuralNode*, NeuralBond*> prevBonds;  //�������ֻ����weight������
	std::map<NeuralNode*, NeuralBond*> nextBonds;  //nextʵ��Ϊprev�ľ���start��end��ͬʱ�������ͬһ��ָ��

	NeuralBond*& getPrevBond(NeuralNode* node) { return prevBonds[node]; };
	NeuralBond*& getNextBond(NeuralNode* node) { return nextBonds[node]; };  

	//double outputValue;
	//double inputValue;
	//double expect;
	
	//��������
	//û���±갲ȫ��飬ʹ�������أ�
	static int dataAmount;
	std::vector<double> outputValues;
	std::vector<double> inputValues;
	std::vector<double> expects;
	std::vector<double> deltas;

	void setExpect(double x, int i = -1);  //�����ڴ�ֵ��һ�����������ڵ�
	void setInput(double x, int i = -1);   //��������ֵ���������ڳ����ڵ�
	void setOutput(double x, int i = -1);  //�������ֵ��һ�����������ڵ�
	double getOutput(int i = -1);

	void collectInputValue();
	void activeOutputValue();
	void active();

	//feedback��active�ĵ���
	std::function<double(double)> activeFunction = ActiveFunctions::sigmoid;
	std::function<double(double)> dactiveFunction = ActiveFunctions::dsigmoid;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive);

	static void connect(NeuralNode* start, NeuralNode* end, double w = 0);
	void connectStart(NeuralNode* node, double w = 0);
	void connectEnd(NeuralNode* node, double w = 0);

	//double delta;
	void updateOneDelta();  //����

	void setVectorValue(std::vector<double>& vec, double x = -1) { for (auto& v : vec) v = x; }

	void setDataAmount(int n);
	int getDataAmount() { return dataAmount; }
	void updateDelta();
	void backPropagate(double learnSpeed = 0.5);

	bool actived, backPropageted;

};

