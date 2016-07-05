#pragma once
#include <random>

class Random
{
public:
	Random();
	virtual ~Random();

	std::random_device rd;
	static std::mt19937 generator;
	std::uniform_real_distribution<double> uniform_dist{ 0, 1 };
	std::normal_distribution<double> normal_dist{ 0, 1 };

	void set_uniform(double a, double b);
	double rand_uniform();
	double rand_normal();

	void reset();

};

