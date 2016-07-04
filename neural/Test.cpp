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

int Test::MNIST_readImageFile(const char* filename, double* input)
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
	memset(input, 0, sizeof(double)*size);
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

int Test::MNIST_readLabelFile(const char* filename, double* expect)
{
	auto content = readFile(filename);
	reverse(content + 4, 4);
	int count = *(int*)(content + 4);
	//fprintf(stderr, "%-30s%d groups data\n", filename, count);
	//expect = new double[count*10];
	memset(expect, 0, sizeof(double)*count * 10);
	for (int i = 0; i < count; i++)
	{
		int pos = *(content + 8 + i);
		expect[i * 10 + pos] = 1;
	}
	delete[] content;
	return 10;
}


void Test::test()
{
	//Matrix::initCuda();
	Matrix A(4, 4, 1, 1);
	A.initRandom();
	A.print();
	Matrix B(2, 2, 1, 1);
	auto m = new int[A.getDataCount()];
	Matrix::poolingForward(re_Average, &A, &B, 2, 2, 2, 2, &m);
	printf("\n");
	B.print();
	for (int i = 0; i < 4; i++)
	{
		printf("%d ", m[i]);
	}
	printf("\n");
	Matrix DA(4, 4, 1, 1);
	Matrix DB(2, 2, 1, 1);
	DB.initInt();
	Matrix::poolingBackward(re_Average, &B, &DB, &A, &DA, 2, 2, 2, 2, m);
	DB.print();
	printf("\n");
	DA.print();

	printf("\nconvolution test:\n");
	//A = Matrix(4,4,2,2);
	Matrix a(4, 4, md_Outside);
	A.initInt();
	Matrix K(2, 2);
	K.initData(1);
	Matrix R(3, 3, 2, 2);
	Matrix r(3, 3, md_Outside);
	Matrix::convolutionForward(&A, &K, &R, 4, 4, 3, 3, 2);
	//A->print(stdout);
	for (int i = 0; i < 4; i++)
	{
		a.resetDataPointer(A.getDataPointer(16 * i, 0));
		a.print();
		r.resetDataPointer(R.getDataPointer(9 * i, 0));
		r.print();
		printf("\n");
	}
	//Matrix::destroyCuda();
	delete m;
}
