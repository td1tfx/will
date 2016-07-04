#pragma once
#include <random>

class Random
{
public:
	Random();
	virtual ~Random();

	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<double> uniform_dist{0,1};
	std::normal_distribution<double> normal_dist{0,1};

	double rand();
	
};

