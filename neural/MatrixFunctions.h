#pragma once
extern "C"
{
#include "cblas.h"
}
#include <stdio.h>

class MatrixFunctions
{
public:
	MatrixFunctions();
	~MatrixFunctions();

	static void d_matrixProduct(double* A,double* B,double* R, int m, int l, int n);
	static void matrixOutput(double* A, int m, int n);
};

