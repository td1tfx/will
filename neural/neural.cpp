#include "NeuralNet.h"
#include "lib/Timer.h"

void test();

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

	test();

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

void test()
{
	// 	auto A = new d_matrix(16, 2);
	// 	//A->initRandom();
	// 	A->print(stdout);
	// 	auto B = new d_matrix(4, 2);
	// 	d_matrix::resample_colasImage(A, B, 2, 4, 1, 2, 2, re_Findmax);
	// 	printf("\n");
	// 	B->print(stdout);

	auto A = new d_matrix(32, 2);
	auto a = new d_matrix(4, 4, 0);
	A->initInt();

	auto K = new d_matrix(2, 2);
	K->initData(1);
	printf("\n");
	auto R = new d_matrix(18, 2);
	auto r = new d_matrix(3, 3, 0);
	
	d_matrix::convolution_colasImage(A, K, R, 4, 4, 3, 3, 2);
	//A->print(stdout);
	a->resetDataPointer(A->getDataPointer(0, 0));
	a->print();
	r->resetDataPointer(R->getDataPointer(0, 0));
	r->print();
	a->resetDataPointer(A->getDataPointer(16, 0));
	a->print();
	r->resetDataPointer(R->getDataPointer(9, 0));
	r->print();
}
