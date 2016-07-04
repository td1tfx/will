#include "Random.h"


Random::Random()
{
	//gen();
	//uniform_dist = std::uniform_real_distribution<double>(0.0,1.0);
}

Random::~Random()
{

}

double Random::rand()
{
	return uniform_dist(gen);
}
