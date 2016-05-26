#pragma once
extern "C"
{
#include "cblas.h"
}
#include <stdio.h>

namespace MatrixFunctions
{
	struct d_matrix
	{
	private:
		double* data = nullptr;
	public:
		int m;
		int n;
		d_matrix(int x, int y) 
		{
			m = x;
			n = y;
			data = new double[m*n];
		}
		~d_matrix()
		{
			delete[] data;
		}
		double& getData(int x, int y)
		{
			return data[x + y*m];
		}
		void output();
	};

	void d_matrixProduct(double* A, double* B, double* R, int m, int k, int n, 
		double a = 1, double c = 0, CBLAS_TRANSPOSE ta = CblasNoTrans, CBLAS_TRANSPOSE tb = CblasNoTrans);
	void matrixOutput(double* A, int m, int n);
	void d_hadamardProduct(double* A, double* B, double* R, int m, int n);
	void d_matrixMinus(double* A, double* B, double* R, int m, int n);
};
