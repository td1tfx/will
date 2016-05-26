#include "MatrixFunctions.h"


//matrix product, R(m*n) = a*A(m*k)*B(k*n) + c*R(m*n)
//call blas
void MatrixFunctions::d_matrixProduct(double* A, double* B, double* R, int m, int k, int n, 
	double a /*= 1*/, double c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/, CBLAS_TRANSPOSE tb /*= CblasNoTrans*/)
{
	int lda = k;
	int ldb = n;
	if (ta == CblasTrans) lda = n;
	if (tb == CblasTrans) ldb = k;
	cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, a, A, lda, B, ldb, c, R, n);
}

//output a matrix
void MatrixFunctions::matrixOutput(double* A, int m, int n)
{

}

//hadamard element wise product
void MatrixFunctions::d_hadamardProduct(double* A, double* B, double* R, int m, int n)
{
	for (int i = 0; i < m*n; i++)
	{
		R[i] = A[i] * B[i];
	}
}

void MatrixFunctions::d_matrixMinus(double* A, double* B, double* R, int m, int n)
{
	for (int i = 0; i < m*n; i++)
	{
		R[i] = A[i] - B[i];
	}
}

void MatrixFunctions::d_matrix::output()
{
	for (int i1 = 0; i1 < m; i1++)
	{
		for (int i2 = 0; i2 < n; i2++)
		{
			printf("%11.5lf ", getData(i1, i2));
		}
		printf("\n");
	}
	printf("\n");
}
