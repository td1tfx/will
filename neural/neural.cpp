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
	std::string filename = "data3.txt";
	net->readData(filename);
	net->setLayers();
	net->train(100000);
	net->test();
	net->outputWeight();
	//double a[2] = { 1, 2 };
	//double b[1] = { 3 };

	//net->activeOutputValue(a,b);

	delete net;
}