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
	printf("Run neural net end. Time is %d ms.\n", clock()-t0);
	
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run_neural(int option)
{
	auto net = new NeuralNet();

	net->readData("p.txt");
	if (option == 0)
		net->createByData(NeuralLayerMode::HaveConstNode, 3, 10);
	else
		net->createByLoad("save0.txt");

	net->setLearnMode(NeuralNetLearnMode::Batch);
	net->setWorkMode(NeuralNetWorkMode::Probability);

	net->setLearnSpeed(0.5);
	//net->selectTest();
	net->train(int(1e5), 1e-3);
	net->test();
	net->outputBondWeight("save.txt");

	delete net;

}
