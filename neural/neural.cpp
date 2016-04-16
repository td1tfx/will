#include "NeuralNet.h"

void run_neural(int option = 0);

int main(int argc, char* argv[])
{
	int option = 1;
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

	net->readData("data2.txt");
	if (option == 0)
		net->createByData();
	else
		net->createByLoad("save2.txt");

	net->setLearnSpeed(0.5);
	//net->selectTest();
	net->train(int(2e5), 1e-3);
	net->test();
	net->outputWeight();

	delete net;

}