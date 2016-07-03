#pragma once
#include "lib/libconvert.h"
#include <string.h>

class Test
{
private:
	static unsigned char* readFile(const char* filename);
	static void reverse(unsigned char* c, int n);
public:
	static int MNIST_readImageFile(const char* filename, double* input);
	static int MNIST_readLabelFile(const char* filename, double* expect);
	static void test();
};

