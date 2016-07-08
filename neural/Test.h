#pragma once
#include "libconvert.h"
#include <string.h>
#include "types.h"

class Test
{
private:
	static unsigned char* readFile(const char* filename);
	static void reverse(unsigned char* c, int n);
public:
	static int MNIST_readImageFile(const char* filename, real* input);
	static int MNIST_readLabelFile(const char* filename, real* expect);
	static void test();
};

