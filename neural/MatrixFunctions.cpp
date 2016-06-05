#include "MatrixFunctions.h"


cublasHandle_t d_matrix::handle;
bool d_matrix::globalUseCublas = false;
bool d_matrix::inited = false;

//注意
void d_matrix::resetDataPointer(double* d)
{
	if (UseCublas)
	{
		memcpyDataIn(d, max_script);
	}
	else
	{
		data = d;
	}
}

void d_matrix::initCublas()
{
	if (inited) { return; }
	inited = true;
#ifdef _USE_CUDA
	int dev = -1;
	auto status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
	}
	dev = findCudaDevice(0, nullptr);
	globalUseCublas = (dev >= 0);
#endif	
}

void d_matrix::print(FILE* fout)
{
	auto temp = malloc_getDataFromDevice();
	for (int i1 = 0; i1 < row; i1++)
	{
		for (int i2 = 0; i2 < col; i2++)
		{
			fprintf(fout, "%14.11lf ", temp[i1 + i2]);
		}
		fprintf(fout, "\n");
	}
	freeDataForDevice(temp);
}

//参数指针必须指向Host内存！
void d_matrix::memcpyDataIn(double* src, int size)
{
	if (UseCublas)
	{
		cudaMemcpy(data, src, int(sizeof(double)*std::min(size, max_script)), cudaMemcpyHostToDevice);
	}
	else
	{
		memcpy(data, src, int(sizeof(double)*std::min(size, max_script)));
	}
}

//参数指针必须指向Host内存！
void d_matrix::memcpyDataOut(double* dst, int size)
{
	if (UseCublas)
	{
		cudaMemcpy(dst, data, int(sizeof(double)*std::min(size, max_script)), cudaMemcpyDeviceToHost);
	}
	else
	{
		memcpy(dst, data, int(sizeof(double)*std::min(size, max_script)));
	}
}

//这两个的操作没有数学道理
//将第一列复制到整个矩阵
void d_matrix::expand()
{
	if (UseCublas)
	{
		for (int i = 1; i < col; i++)
		{
			cudaMemcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(double)*row, cudaMemcpyDeviceToDevice);
		}
	}
	else
	{
		//#pragma loop(hint_parallel(8))
		for (int i = 1; i < col; i++)
		{
			memcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(double)*row);
		}
	}
}

int d_matrix::indexColMaxAbs(int c)
{
	if (UseCublas)
	{
		int r;
		cublasIdamax(handle, row, getDataPointer(0, c), 1, &r);
		return r - 1;
	}
	else
	{
		return cblas_idamax(row, getDataPointer(0, c), 1);
	}
}

double d_matrix::sumColAbs(int c)
{
	if (UseCublas)
	{
		double r;
		cublasDasum(handle, row, getDataPointer(0, c), 1, &r);
		return r;
	}
	else
	{
		return cblas_dasum(row, getDataPointer(0, c), 1);
	}
}

double d_matrix::ddot()
{
	if (UseCublas)
	{
		double r;
		cublasDdot(handle, max_script, data, 1, data, 1, &r);
		return r;
	}
	else
	{
		return cblas_ddot(max_script, data, 1, data, 1);
	}
}

void d_matrix::initData(double v)
{
	auto temp = mallocDataForDevice();
#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		temp[i] = v;
	}
	set_freeDataToDevice(temp);
}

//注意这个函数调用次数很少
void d_matrix::initRandom()
{
	auto temp = mallocDataForDevice();
	//#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		temp[i] = 2.0 * rand() / RAND_MAX - 1;
	}
	set_freeDataToDevice(temp);
}


void d_matrix::multiply(double v)
{
	if (UseCublas)
	{
		cublasDscal(handle, row, &v, data, 1);
	}
	else
	{
		cblas_dscal(max_script, v, data, 1);
	}
}

void d_matrix::colMultiply(double v, int c)
{
	if (UseCublas)
	{
		cublasDscal(handle, row, &v, getDataPointer(0, c), 1);
	}
	else
	{
		cblas_dscal(row, v, getDataPointer(0, c), 1);
	}
}

void d_matrix::applyFunction(std::function<double(double)> f)
{
	applyFunction(this, this, f);
}


//复制数据，只处理较少的
void d_matrix::cpyData(d_matrix* dst, d_matrix* src)
{
	if (dst->UseCublas)
	{
		cudaMemcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col), cudaMemcpyDeviceToDevice);
	}
	else
	{
		memcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col));
	}
}

void d_matrix::product(d_matrix* A, d_matrix* B, d_matrix* R,
	double a /*= 1*/, double c /*= 0*/, d_matrixTrans ta /*= NoTrans*/, d_matrixTrans tb /*= NoTrans*/)
{
	int m = R->row;
	int n = R->col;
	int lda = A->row;
	int k = A->col;
	int ldb = B->row;
	if (ta == Trans) { k = A->row; }
	if (globalUseCublas)
	{
		auto ta1 = get_cublas_trans(ta);
		auto tb1 = get_cublas_trans(tb); 
		cublasDgemm(handle, ta1, tb1, m, n, k, &a, A->data, lda, B->data, ldb, &c, R->data, m);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta); 
		auto tb1 = get_cblas_trans(tb);
		cblas_dgemm(CblasColMajor, ta1, tb1, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
	}
}

void d_matrix::productVector(d_matrix* A, d_matrix* B, d_matrix* R, double a /*= 1*/, double c /*= 0*/, d_matrixTrans ta /*= NoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == Trans) { std::swap(m, n); };

	if (globalUseCublas)
	{
		auto ta1 = get_cublas_trans(ta);
		cublasDgemv(handle, ta1, m, n, &a, A->data, A->row, B->data, 1, &c, R->data, 1);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta); 
		cblas_dgemv(CblasColMajor, ta1, m, n, a, A->data, A->row, B->data, 1, c, R->data, 1);
	}
}

void d_matrix::hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R)
{
	auto tempA = A->malloc_getDataFromDevice();
	auto tempB = B->malloc_getDataFromDevice();
	auto tempR = R->mallocDataForDevice();
#pragma loop(hint_parallel(8))
	for (int i = 0; i < R->max_script; i++)
	{
		tempR[i] = tempA[i] * tempB[i];
	}
	A->freeDataForDevice(tempA);
	B->freeDataForDevice(tempB);
	R->set_freeDataToDevice(tempR);
}

void d_matrix::minus(d_matrix* A, d_matrix* B, d_matrix* R)
{
	if (globalUseCublas)
	{
		double a = -1;
		cublasDcopy(handle, R->max_script, A->data, 1, R->data, 1);
		cublasDaxpy(handle, R->max_script, &a, B->data, 1, R->data, 1);
	}
	else
	{
		cblas_dcopy(R->max_script, A->data, 1, R->data, 1);
		cblas_daxpy(R->max_script, -1, B->data, 1, R->data, 1);
	}
// #pragma loop(hint_parallel(8))
// 	for (int i = 0; i < R->max_script; i++)
// 	{
// 		R->data[i] = A->data[i] - B->data[i];
// 	}
}

void d_matrix::applyFunction(d_matrix* A, d_matrix* R, std::function<double(double)> f)
{
	auto tempA = A->malloc_getDataFromDevice();
	auto tempR = tempA;
	if (A != R)
		tempR = R->mallocDataForDevice();
#pragma loop(hint_parallel(8))
	for (int i = 0; i < std::min(A->max_script, R->max_script); i++)
	{
		tempR[i] = f(tempA[i]);
	}
	if (A != R)
		A->set_freeDataToDevice(tempA);
	R->set_freeDataToDevice(tempR);
}

double* d_matrix::mallocData(int size)
{
	if (UseCublas)
	{
		double* d;
		if (cudaMalloc((void **)&d, size * sizeof(double)) == cudaSuccess)
		{
			dataIsWhere = DataInDevice;
			return d;
		}
	}
	return new double[size];
}

void d_matrix::freeData()
{
	if (UseCublas)
	{
		if (dataIsWhere = DataInDevice)
		{
			cudaFree(data);
			return;
		}
	}
	delete data;
}

double* d_matrix::malloc_getDataFromDevice()
{
	if (UseCublas)
	{
		auto temp = new double[max_script];
		cudaMemcpy(temp, data, sizeof(double)*max_script, cudaMemcpyDeviceToHost);
		return temp;
	}
	else
	{
		return data;
	}
}

void d_matrix::freeDataForDevice(double* temp)
{
	if (UseCublas)
	{
		delete temp;
	}
}

double* d_matrix::mallocDataForDevice()
{
	if (UseCublas)
	{
		return new double[max_script];
	}
	else
	{
		return data;
	}
}

void d_matrix::set_freeDataToDevice(double* temp)
{
	if (UseCublas)
	{
		cudaMemcpy(data, temp, sizeof(double)*max_script, cudaMemcpyHostToDevice);
		delete temp;
	}
}

