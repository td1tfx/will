#include "NeuralNet.h"
#include <time.h>

void run_neural(int option = 0);

int main(int argc, char* argv[])
{
	int option = 0;
	if (argc > 1)
	{
		option = atoi(argv[1]);
	}
	clock_t t0 = clock();
	run_neural(option);
	printf("Run neural net end. Time is %d ms.\n", clock() - t0);
	
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run_neural(int option)
{
	auto net = new NeuralNet();
	
	net->setLearnMode(NeuralNetLearnMode::Batch);
	net->setWorkMode(NeuralNetWorkMode::Fit);

	net->readData("test.txt");
	if (option == 0)
		net->createByData(NeuralLayerMode::HaveConstNode, 3, 30);
	else
		net->createByLoad("save.txt");

	net->setLearnSpeed(0.1);
	net->selectTest();
	net->train(int(1e6), 1e-4);
	net->test();
	net->outputBondWeight("save.txt");

	delete net;

}