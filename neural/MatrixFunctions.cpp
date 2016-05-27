#include "MatrixFunctions.h"


void d_matrix::print()
{
	for (int i1 = 0; i1 < m; i1++)
	{
		for (int i2 = 0; i2 < n; i2++)
		{
			printf("%11.5lf ", getData(i1, i2));
		}
		printf("\n");
	}
// 	for (int i = 0; i < m*n; i++)
// 	{
// 			printf("%11.5lf ", getData(i));
// 	}
 	printf("\n");
}

void d_matrix::product(d_matrix* A, d_matrix* B, d_matrix* R, 
	double a /*= 1*/, double c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/, CBLAS_TRANSPOSE tb /*= CblasNoTrans*/)
{
	int m = R->m;
	int n = R->n;
	int lda = A->m;
	int k = A->n;
	int ldb = B->m;
	
	if (ta == CblasTrans) 
	{
		k = A->m;
		//lda = k;		
	}
	if (tb == CblasTrans)
	{
		//ldb = n;
	}
	cblas_dgemm(CblasColMajor, ta, tb, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
}

void d_matrix::hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R)
{
	for (int i = 0; i < R->m*R->n; i++)
	{
		R->data[i] = A->data[i] * B->data[i];
	}
}

void d_matrix::minus(d_matrix* A, d_matrix* B, d_matrix* R)
{
	for (int i = 0; i < R->m*R->n; i++)
	{
		R->data[i] = A->data[i] - B->data[i];
	}
}
