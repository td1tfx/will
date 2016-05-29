#pragma once
#include "libconvert.h"

class MNISTFunctions
{
public:
	MNISTFunctions();
	virtual ~MNISTFunctions();

	static unsigned char* readFile(const char* filename);
	static int readImageFile(const char* filename, double*& input);
	static int readLabelFile(const char* filename, double*& expect);
};

