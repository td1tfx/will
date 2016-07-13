#include "Test.h"
#include "Matrix.h"

unsigned char* Test::readFile(const char* filename)
{
	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Can not open file %s\n", filename);
		return nullptr;
	}
	fseek(fp, 0, SEEK_END);
	int length = ftell(fp);
	fseek(fp, 0, 0);
	auto s = new unsigned char[length + 1];
	for (int i = 0; i <= length; s[i++] = '\0');
	fread(s, length, 1, fp);
	fclose(fp);
	return s;
}

void Test::reverse(unsigned char* c, int n)
{
	for (int i = 0; i < n / 2; i++)
	{
		auto& a = *(c + i);
		auto& b = *(c + n - 1 - i);
		auto t = b;
		b = a;
		a = t;
	}
}

int Test::MNIST_readImageFile(const char* filename, real* input)
{
	auto content = readFile(filename);
	reverse(content + 4, 4);
	reverse(content + 8, 4);
	reverse(content + 12, 4);
	int count = *(int*)(content + 4);
	int w = *(int*)(content + 8);
	int h = *(int*)(content + 12);
	//fprintf(stderr, "%-30s%d groups data, w = %d, h = %d\n", filename, count, w, h);
	int size = count*w*h;
	//input = new double[size];
	memset(input, 0, sizeof(real)*size);
	for (int i = 0; i < size; i++)
	{
		auto v = *(content + 16 + i);
		input[i] = v / 255.0;
	}

	// 	int check = 59990;
	//  	for (int i = 784 * check; i < 784*(check+10); i++)
	// 	{
	// 		if (input[i] != 0)
	// 			printf("1", input[i]);
	// 		else
	// 			printf(" ");
	// 		if (i % 28 == 27)
	// 			printf("\n");
	// 	}

	delete[] content;
	return w*h;
}

int Test::MNIST_readLabelFile(const char* filename, real* expect)
{
	auto content = readFile(filename);
	reverse(content + 4, 4);
	int count = *(int*)(content + 4);
	//fprintf(stderr, "%-30s%d groups data\n", filename, count);
	//expect = new double[count*10];
	memset(expect, 0, sizeof(real)*count * 10);
	for (int i = 0; i < count; i++)
	{
		int pos = *(content + 8 + i);
		expect[i * 10 + pos] = 1;
	}
	delete[] content;
	return 10;
}




void Test::testSoftmax(int tests)
{
	if (tests)
	{
		Matrix X(4, 4), A(4, 4);
		Matrix dX(4, 4), dA(4, 4);
		Matrix E(4, 4, md_Inside, mc_NoCuda);
		E.initData(0);
		for (int i = 0; i < 4; i++) E.getData(i, i) = 1;
		E.tryUploadToCuda();
		E.print();

		dX.initData(1);
		X.initData(1);
		X.initRandom();
		Matrix::activeForward(af_SoftmaxLoss, &X, &A);
		Matrix::add(&E, -1, &A, &dA);

		printf("Y:\n");
		A.print();
		printf("dY:\n");
		dA.print();
		printf("X:\n");
		X.print();

		Matrix::activeBackward(af_SoftmaxLoss, &A, &dA, &X, &dX);
		printf("dX:\n");
		dX.print();
	}
}

void Test::testConvolution(int testc)
{
	if (testc)
	{
		printf("\nconvolution test:\n");
		int c = 1;
		int n = 2;
		int kc = 1;
		Matrix X(4, 4, 1, n);
		Matrix dX(4, 4, 1, n);

		Matrix W(2, 2, 1, 1);
		Matrix dW(2, 2, 1, 1);

		Matrix A(3, 3, 1, n);
		Matrix dA(3, 3, 1, n);
		Matrix dB(1, 1, 1, 1);

		A.initData(0);
		dX.initData(12);
		X.initInt(0);
		W.initData(1);
		dA.initRandom();
		Matrix::convolutionForward(&X, &W, &A);

		printf("X\n");
		X.print();
		printf("W\n");
		W.print();
		printf("A\n");
		A.print();
		printf("\n");
		printf("dA\n");
		dA.print();

		Matrix::convolutionBackward(&A, &dA, &X, &dX, &W, &dW, &dB);
		printf("dX\n");
		dX.print();
		printf("dW\n");
		dW.print();
		printf("dB\n");
		dB.print();

		Matrix a(4, 4, md_Outside);
		Matrix r(3, 3, md_Outside);
		for (int i = 0; i < c * n; i++)
		{
			// 			a.resetDataPointer(X.getDataPointer(i * 16));
			// 			a.print();
			// 			r.resetDataPointer(A.getDataPointer(i * 9));
			// 			r.print();
			// 			printf("\n");
		}
	}
}

void Test::testPooling(int testp)
{
	if (testp)
	{
		printf("\npooling test:\n");
		Matrix X(3, 3, 1, 1);
		X.initRandom();
		X.print();
		Matrix A(2, 2, 1, 1);
		auto m = new int[X.getDataCount()];
		auto re = re_Average_Padding;
		Matrix::poolingForward(re, &X, &A, 2, 2, 2, 2, m);
		printf("\n");
		A.print();
		for (int i = 0; i < X.getDataCount(); i++)
		{
			//printf("%d ", m[i]);
		}
		printf("\n");
		Matrix dX(3, 3, 1, 1);
		Matrix dA(2, 2, 1, 1);
		dA.initInt();
		Matrix::poolingBackward(re, &A, &dA, &X, &dX, 2, 2, 2, 2, m);
		dA.print();
		printf("\n");
		dX.print();
		delete m;
	}
}

void Test::test()
{
	testPooling(0);
	testConvolution(1);
	testSoftmax(0);
}

void Test::test2()
{
	Matrix::initCuda();
	printf("Use Cuda\n");
	test();
	Matrix::destroyCuda();
	printf("No Cuda\n");
	test();
}
