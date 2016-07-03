#include "NeuralNet.h"
#include "lib/Timer.h"

int main(int argc, char* argv[])
{
	NeuralNet net;
	Timer t;

	if (argc > 1)
	{
		net.loadOption(argv[1]);
	}
	else
	{
		net.loadOption("p.ini");
	}

	t.start();
	net.run();
	t.stop();
	Test::test();

	fprintf(stdout, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());

#ifdef _WIN32
	fprintf(stderr, "\nPress any key to exit.\n");
	getchar();
#endif
	return 0;
}
