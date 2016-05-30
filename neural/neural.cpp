#include "NeuralNet.h"
#include <time.h>

void run_neural(int option);

int main(int argc, char* argv[])
{
	int option = 0;
	if (argc > 1)
	{
		option = atoi(argv[1]);
	}
	clock_t t0 = clock();
	run_neural(option);
	clock_t t = clock() - t0;
	fprintf(stderr, "Run neural net end. Time is %lf s.\n", double(t) / CLOCKS_PER_SEC);
	
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run_neural(int option)
{
	auto net = new NeuralNet();

	net->readData("p.txt");
	//net->readMNIST();

	if (option == 0)
		net->createByData(3, 7);
	else
		net->createByLoad("save.txt");

	net->setLearnMode(NeuralNetLearnMode::Batch);
	net->setWorkMode(NeuralNetWorkMode::Probability);

	net->setLearnSpeed(0.5);
	net->setRegular(0.01);
	//net->selectTest();
	net->train(int(1e5), 1000, 1e-3, 1e-9);
	net->test();
	net->outputBondWeight("save.txt");

	delete net;

}
