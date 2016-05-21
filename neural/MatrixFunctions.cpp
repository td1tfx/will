#include "MatrixFunctions.h"



MatrixFunctions::MatrixFunctions()
{
}


MatrixFunctions::~MatrixFunctions()
{
}

//R(m*n) = A(m*l) * B(l*n)
void MatrixFunctions::d_matrixProduct(double* A, double* B, double* R, int m, int l, int n)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, l, 1.0, A, l, B, n, 0.0, R, n);
}

void MatrixFunctions::matrixOutput(double* A, int m, int n)
{
	for (int i1 = 0; i1 < m; i1++)
	{
		for (int i2 = 0; i2 < n; i2++)
		{
			printf("%lf ", A[i2 + (i1 - 1)*m]);
		}
		printf("\n");
	}
}
