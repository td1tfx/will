#include "NeuralNet.h"

void run_neural();

int option = 0;

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
	
	net->readData("data2.txt");
	//net->createByData();
	net->createByLoad("save2.txt");
	net->setLearnSpeed(0.5);
	//net->selectTest();
	net->train(int(1e6), 1e-3);
	//net->test();
	net->outputWeight();

	delete net;
}