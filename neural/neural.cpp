#include "NeuralNet.h"

void run();

int main(int argc, char* argv[])
{
	run();
	printf("run end.\n");
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run()
{
	auto net = new NeuralNet();
	std::string filename = "data.txt";
	net->readData(filename);
	net->setLayers();
	net->selectTest();
	net->train(200000,1e-4);
	net->test();
	net->outputWeight();
	//double a[2] = { 1, 2 };
	//double b[1] = { 3 };

	//net->activeOutputValue(a,b);

	delete net;
}