#include "Random.h"

std::mt19937 Random::generator;

Random::Random()
{
	//gen();
	//uniform_dist = std::uniform_real_distribution<double>(0.0,1.0);
}

Random::~Random()
{

}

void Random::set_uniform(real a, real b)
{
	uniform_dist = std::uniform_real_distribution<real>(a, b);
}

real Random::rand_uniform()
{
	return uniform_dist(generator);
}

real Random::rand_normal()
{
	return normal_dist(generator);
}

void Random::reset()
{
	generator = std::mt19937(rd());
}
