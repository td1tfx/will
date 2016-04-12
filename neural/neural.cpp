#include "NeuralNet.h"

void run();

int main(int argc, char* argv[])
{
	run();
#ifdef _DEBUG
	printf("debug end.\n");
	getchar();
#endif
	return 0;
}

void run()
{
	auto net = new NeuralNet();
	std::string filename = "data.txt";
	net->readData(filename);
	net->createLayers(3);
	net->setLayers();

	double a[2] = { 1, 2 };
	double b[1] = { 3 };

	net->calOutput(a,b);

	printf("output = %lf\n",b[0]);

	delete net;
}