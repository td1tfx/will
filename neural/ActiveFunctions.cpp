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

double ActiveFunctions::sigmoid(double input)
{
	return 1.0 / (1 + exp(input));
}

double ActiveFunctions::_sigmoid(double output)
{
	return 1.0 / (1 + exp(output));
}
