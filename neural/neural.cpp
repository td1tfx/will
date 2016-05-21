#include "NeuralNet.h"

using namespace MatrixFunctions;

void run_neural(int option = 0);

int main(int argc, char* argv[])
{
	int option = 0;
	if (argc > 1)
	{
		int option = atoi(argv[1]);
	}
	//run_neural(option);
	printf("Run neural net end.\n");

	double A[12], B[12], C[16];
	for (int i = 0; i < 12; i++)
	{
		A[i] = i;
		B[i] = i;
	}
	matrixOutput(A, 4, 3);
	matrixOutput(A, 3, 4);
	d_matrixProduct(A,A,C,4,3,4);
	matrixOutput(C, 4, 4);

	
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

void run_neural(int option)
{
	auto net = new NeuralNet();

	net->readData("test.txt");
	if (option == 0)
		net->createByData(NeuralLayerMode::HaveConstNode, 3, 7);
	else
		net->createByLoad("save2.txt");

	net->setLearnSpeed(0.001);
	net->setLearnMode(NeuralNetLearnMode::Batch);
	net->setWorkMode(NeuralNetWorkMode::Fit);
	net->selectTest();
	net->train(int(1e6), 1e-4);
	net->test();
	//net->outputBondWeight("save.txt");

	delete net;

}