#include "NeuralNet.h"

void run_neural();

int main(int argc, char* argv[])
{
	run_neural();
	printf("run end.\n");
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run_neural()
{
	auto net = new NeuralNet();
	std::string filename = "data3.txt";
	
	net->readData(filename);
	net->setLayers(0.01);
	net->selectTest();
	net->train(int(1e5), 1e-3);
	net->test();
	net->outputWeight();

	delete net;
}