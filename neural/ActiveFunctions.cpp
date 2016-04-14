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

double ActiveFunctions::sigmoid(double x)
{
	return 1.0 / (1 + exp(-x));
}

double ActiveFunctions::dsigmoid(double x)
{
	double a = 1 + exp(-x);
	return exp(-x) / (a*a);
}
