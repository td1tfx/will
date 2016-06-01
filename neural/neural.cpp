#include "NeuralNet.h"
#include "lib/Timer.h"

int main(int argc, char* argv[])
{
	NeuralNet net;
	Timer t;

	if (argc > 1)
	{
		net.loadOptoin(argv[1]);
	}
	else
		net.loadOptoin("p.ini");
	
	t.start();
	net.run();
	t.stop();
	
	fprintf(stderr, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());

#ifdef _WIN32
	getchar();
#endif
	return 0;
}


