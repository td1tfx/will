#pragma once
extern "C"
{
#include "cblas.h"
}
#include <stdio.h>

namespace MatrixFunctions
{
	void d_matrixProduct(double* A,double* B,double* R, int m, int l, int n);
	void matrixOutput(double* A, int m, int n);
	void hadamardProduct(double* A, double* B, double* R, int m, int n);
};
