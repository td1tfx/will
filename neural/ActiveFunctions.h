#pragma once
#include <math.h>

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

	static double sigmoid(double input);
	static double _sigmoid(double output);
	
};

