#include "NeuralNet.h"

void run_neural(int option = 0);

int main(int argc, char* argv[])
{
	int option = 0;
	if (argc > 1)
	{
		int option = atoi(argv[1]);
	}
	run_neural(option);
	printf("Run neural net end.\n");
	
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

	net->readData("p1.txt");
	if (option == 0)
		net->createByData(NeuralLayerMode::HaveConstNode, 3, 10);
	else
		net->createByLoad("save.txt");

	net->setLearnSpeed(0.1);
	net->selectTest();
	net->train(int(1e5), 1e-6);
	net->test();
	net->outputBondWeight("save.txt");

	delete net;

}