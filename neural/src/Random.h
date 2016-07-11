#pragma once
#include <random>

template <typename T = float>
class Random
{
public:
	Random() {}
	virtual ~Random() {}

	std::random_device rd;
	std::mt19937 generator;
	std::uniform_real_distribution<T> uniform_dist{ 0, 1 };
	std::normal_distribution<T> normal_dist{ 0, 1 };

	void set_uniform(T a, T b)
	{
		uniform_dist = std::uniform_real_distribution<T>(a, b);
	}
	T rand_uniform()
	{
		return uniform_dist(generator);
	}
	T rand_normal()
	{
		return normal_dist(generator);
	}
	void reset()
	{
		generator = std::mt19937(rd());
	}

};

