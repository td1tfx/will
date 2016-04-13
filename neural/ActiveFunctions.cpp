#include "ActiveFunctions.h"



ActiveFunctions::ActiveFunctions()
{
}


ActiveFunctions::~ActiveFunctions()
{
}

// void ActiveFunctions::setFunctions(NeuralNode* node, std::function<double(double)> activeFunction, std::function<double(double)> feedbackFunction)
// {
// 	node->activeFunction = activeFunction;
// 	node->feedbackFunction = feedbackFunction;
// }

double ActiveFunctions::sigmoid(double v)
{
	return 1.0 / (1 + exp(-v));
}

double ActiveFunctions::dsigmoid(double v)
{
	double a = 1 + exp(-v);
	return exp(-v) / (a*a);
}
