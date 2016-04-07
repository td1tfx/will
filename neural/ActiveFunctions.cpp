#include "ActiveFunctions.h"



ActiveFunctions::ActiveFunctions()
{
}


ActiveFunctions::~ActiveFunctions()
{
}

double ActiveFunctions::sigmoid(double input)
{
	return 1.0 / (1 + exp(input));
}

double ActiveFunctions::_sigmoid(double output)
{
	return 1.0 / (1 + exp(output));
}
