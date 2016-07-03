#include "NeuralNet.h"
#include "lib/Timer.h"

void test();

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
	test();

	fprintf(stdout, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());

#ifdef _WIN32
	fprintf(stderr, "\nPress any key to exit.\n");
	getchar();
#endif
	return 0;
}

void test()
{
	auto A = new Matrix(4, 4, 1, 3);
	auto m = new int[4];
	A->initRandom();
	A->print();
	auto B = new Matrix(2, 2, 1, 3);
	//Matrix::resample(A, B, re_Max, &m, 0);
	Matrix::resample_colasImage(A, B, 4, 4, 2, 2, 3, re_Max, nullptr);
	printf("\n");
	B->print();
	for (int i = 0; i < 4; i++)
	{
		printf("%d ", m[i]);
	}

	A = new Matrix(32, 2);
	auto a = new Matrix(4, 4, md_Outside);
	A->initInt();
	auto K = new Matrix(2, 2);
	K->initData(1);
	printf("\nconvolution\n");
	auto R = new Matrix(18, 2);
	auto r = new Matrix(3, 3, md_Outside);

	Matrix::convolution_colasImage(A, K, R, 4, 4, 3, 3, 2);
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
