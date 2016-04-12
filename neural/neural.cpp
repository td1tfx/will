#include "NeuralNet.h"

void run();

int main(int argc, char* argv[])
{
	run();
#ifdef _DEBUG
	getchar();
#endif
	return 0;
}

void run()
{
	auto net = new NeuralNet();
	std::string filename = "data.txt";
	net->readData(filename);
	net->createLayers(3);
	delete net;
}