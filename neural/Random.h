#pragma once
#include <random>
#include "types.h"

class Random
{
public:
	Random();
	virtual ~Random();

	std::random_device rd;
	static std::mt19937 generator;
	std::uniform_real_distribution<real> uniform_dist{ 0, 1 };
	std::normal_distribution<real> normal_dist{ 0, 1 };

	void set_uniform(real a, real b);
	real rand_uniform();
	real rand_normal();

	void reset();

};

