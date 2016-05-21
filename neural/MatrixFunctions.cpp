#include "MatrixFunctions.h"


//matrix product, R(m*n) = A(m*l) * B(l*n)
//call blas
void MatrixFunctions::d_matrixProduct(double* A, double* B, double* R, int m, int l, int n)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, l, 1.0, A, l, B, n, 0.0, R, n);
}

//output a matrix to debug
void MatrixFunctions::matrixOutput(double* A, int m, int n)
{
	for (int i1 = 0; i1 < m; i1++)
	{
		for (int i2 = 0; i2 < n; i2++)
		{
			printf("%11.5lf ", A[i2 + i1*n]);
		}
		printf("\n");
	}
	printf("\n");
}

//hadamard element wise product
void MatrixFunctions::hadamardProduct(double* A, double* B, double* R, int m, int n)
{
	for (int i = 0; i < m*n; i++)
	{
		R[i] = A[i] * B[i];
	}
}
