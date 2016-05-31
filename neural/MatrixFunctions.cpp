#include "MatrixFunctions.h"


double d_matrix::ddot()
{
	return cblas_ddot(m*n, data, 1, data, 1);
}

void d_matrix::print()
{
#ifdef _DEBUG
	for (int i1 = 0; i1 < m; i1++)
	{
		for (int i2 = 0; i2 < n; i2++)
		{
			fprintf(stderr, "%11.5lf ", getData(i1, i2));
		}
		fprintf(stderr, "\n");
	}
	// 	for (int i = 0; i < m*n; i++)
	// 	{
	// 			printf("%11.5lf ", getData(i));
	// 	}
	fprintf(stderr, "\n");
#endif
}

void d_matrix::memcpyDataIn(double* src, int size)
{
	memcpy(data, src, std::min(size, int(sizeof(double)*m*n)));
}

void d_matrix::memcpyDataOut(double* dst, int size)
{
	memcpy(dst, data, std::min(size, int(sizeof(double)*m*n)));
}

//这两个的操作没有数学道理
//将第一列复制到整个矩阵
void d_matrix::expand()
{
	for (int i = 1; i < n; i++)
	{
		memcpy(getDataPointer(0,i), getDataPointer(0,0), sizeof(double)*m);
	}
}

int d_matrix::indexRowMaxAbs(int r)
{
	int i = cblas_idamax(n, &getData(r, 0), 1);
	return i - 1;
}

void d_matrix::initData(double v)
{
	for (int i = 0; i < m*n; i++)
	{
		data[i] = v;
	}
}

void d_matrix::initRandom()
{
	for (int i = 0; i < m*n; i++)
	{
		data[i] = 2.0 * rand() / RAND_MAX - 1;
	}
}

void d_matrix::multiply(double v)
{
	for (int i = 0; i < m*n; i++)
	{
		data[i] *= v;
	}
}

//复制数据，只处理较少的
void d_matrix::cpyData(d_matrix* dst, d_matrix* src)
{
	memcpy(dst->data, src->data, sizeof(double)*std::min(dst->m*dst->n, src->m*src->n));
}

void d_matrix::product(d_matrix* A, d_matrix* B, d_matrix* R,
	double a /*= 1*/, double c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/, CBLAS_TRANSPOSE tb /*= CblasNoTrans*/)
{
	int m = R->m;
	int n = R->n;
	int lda = A->m;
	int k = A->n;
	int ldb = B->m;
	if (ta == CblasTrans) { k = A->m; }
	cblas_dgemm(CblasColMajor, ta, tb, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
}

void d_matrix::productVector(d_matrix* A, d_matrix* B, d_matrix* R, double a /*= 1*/, double c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/)
{
	int m = A->m, n = A->n;
	if (ta == CblasTrans) { std::swap(m, n); };
	cblas_dgemv(CblasColMajor, ta, m, n, a, A->data, A->m, B->data, 1, c, R->data, 1);
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
