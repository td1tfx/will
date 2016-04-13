#pragma once
#include <math.h>
//#include "NeuralNode.h"

class ActiveFunctionPair 
{
public:
	virtual double active(double input);
	virtual double _active(double output);
};

class ActiveFunctions
{
public:
	ActiveFunctions();
	virtual ~ActiveFunctions();
	//static void setFunctions(class NeuralNode* node, std::function<double(double)> activeFunction, std::function<double(double)> feedbackFunction);

	double c = 0, f = 0;
	static double sigmoid(double v);
	static double dsigmoid(double v);
	static double linear(double v) { return v; }
};

