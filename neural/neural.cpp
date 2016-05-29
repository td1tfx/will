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
	fprintf(stderr, "Run neural net end. Time is %d ms.\n", clock() - t0);
	
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run_neural(int option)
{
	auto net = new NeuralNet();

	//net->readData("p.txt");
	net->readMNIST();

	if (option == 0)
		net->createByData(3, 100);
	else
		net->createByLoad("save0.txt");

	net->setLearnMode(NeuralNetLearnMode::Batch);
	net->setWorkMode(NeuralNetWorkMode::Probability);

	net->setLearnSpeed(0.5);
	net->setRegular(0.01);
	//net->selectTest();
	net->train(int(1e3), 10, 1e-6, 0);
	net->test();
	net->outputBondWeight("save.txt");

	delete net;

}
