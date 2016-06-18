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
	{
		net.loadOptoin("p.ini");
	}

	auto A = new d_matrix(16, 2);
	//A->initRandom();
	A->print(stdout);
	auto B = new d_matrix(4, 2);
	d_matrix::resample_colasImage(A,B,2,4, 1, 2,2,re_Findmax);
	printf("\n");
	B->print(stdout);

	t.start();
	net.run();
	t.stop();
	
	fprintf(stdout, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());

#ifdef _WIN32
	fprintf(stderr, "\nPress any key to exit.\n");
	getchar();
#endif
	return 0;
}


