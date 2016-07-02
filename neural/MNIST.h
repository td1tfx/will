#pragma once
#include "lib/libconvert.h"
#include <string.h>

class MNIST
{
public:
	static unsigned char* readFile(const char* filename);
	static int readImageFile(const char* filename, double* input);
	static int readLabelFile(const char* filename, double* expect);
	static void reverse(unsigned char* c, int n);
};

