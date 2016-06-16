#include "MatrixFunctions.h"


cublasHandle_t d_matrix::handle;
int d_matrix::globalUseCuda = 0;
bool d_matrix::inited = false;

d_matrix::d_matrix(int x, int y, int tryInsideData /*= 1*/, int tryUseCuda /*= 1*/)
{
	insideData = tryInsideData;
	UseCuda = tryUseCuda && globalUseCuda;

	row = x;
	col = y;
	max_script = row*col;
	if (insideData)
	{
		data = mallocData(max_script);
		data_size = max_script;
	}
}

//返回值：-1空矩阵，未重新分配内存，1重新分配内存
int d_matrix::resize(int m, int n, int force /*= 0*/)
{
	if (!this) 
		return -1;
	row = m;
	col = n;
	max_script = m*n;
	//空间不够或者强制则重新分配
	if (max_script > data_size || force)
	{
		//重新申请空间
		if (insideData)
		{
			freeData();
			data = mallocData(row*col);
		}
		data_size = row*col;
		return 1;
	}
	return 0;
}

//注意，比较危险
void d_matrix::resetDataPointer(double* d, int d_in_cuda /*= 0*/)
{
	if (UseCuda)
	{
		if (d_in_cuda == 0)
		{
			memcpyDataIn(d, max_script);
		}
		else
		{
			data = d;
		}
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
	globalUseCuda = (dev >= 0);
#endif	
}

void d_matrix::print(FILE* fout)
{
	auto temp = malloc_getDataFromDevice();
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			fprintf(fout, "%14.11lf ", temp[xy2i(i, j)]);
		}
		fprintf(fout, "\n");
	}
	freeDataForDevice(temp);
}

int d_matrix::load(double* v, int n)
{
	auto temp = mallocDataForDevice();
	int k = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp[xy2i(i, j)] = v[k++];
			if (k >= n) return k;
		}
	}
	set_freeDataToDevice(temp);
	return k;
}

//参数指针必须指向Host内存！
void d_matrix::memcpyDataIn(double* src, int size)
{
	if (UseCuda)
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
	if (UseCuda)
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
	if (UseCuda)
	{
		for (int i = 1; i < col; i*=2)
		{
			cudaMemcpy(getDataPointer(0, i), getDataPointer(0, 0), 
				sizeof(double)*row*std::min(i, col - i), cudaMemcpyDeviceToDevice);
		}
	}
	else
	{
		//#pragma loop(hint_parallel(8))
		for (int i = 1; i < col; i*=2)
		{
			memcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(double)*row*std::min(i, col-i));
		}
	}
}

int d_matrix::indexColMaxAbs(int c)
{
	if (UseCuda)
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
	if (UseCuda)
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
	if (UseCuda)
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
	if (UseCuda)
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
	if (UseCuda)
	{
		cublasDscal(handle, row, &v, getDataPointer(0, c), 1);
	}
	else
	{
		cblas_dscal(row, v, getDataPointer(0, c), 1);
	}
}


//复制数据，只处理较少的
void d_matrix::cpyData(d_matrix* dst, d_matrix* src)
{
	if (dst->UseCuda)
	{
		cudaMemcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col), cudaMemcpyDeviceToDevice);
	}
	else
	{
		memcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col));
	}
}

void d_matrix::tryUploadToCuda()
{
	if (globalUseCuda)
	{
		if (UseCuda == 0)
		{
			UseCuda = 1;
			auto temp = mallocData(data_size);
			if (temp)
			{
				std::swap(temp, data);
				set_freeDataToDevice(temp);
			}
			else
			{
				UseCuda = 0;
			}
		}
	}
}

void d_matrix::tryDownloadFromCuda()
{
	if (UseCuda == 1)
	{
		auto temp = malloc_getDataFromDevice();
		if (temp)
		{
			std::swap(temp, data);
			cudaFree(temp);
		}
		UseCuda = 0;
	}
}

void d_matrix::shareData(d_matrix* A, int m, int n)
{
	if (!insideData && 
		((UseCuda && A->UseCuda)
		|| (!UseCuda && !A->UseCuda)))
	this->data = A->getDataPointer(m, n);
	/*
	else if (UseCuda && !A->UseCuda)
	{
		memcpyDataIn(A->getDataPointer(m, n), max_script);
	}
	else
	{
	}
	*/
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
	if (globalUseCuda)
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

	if (globalUseCuda)
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

void d_matrix::productVector2(d_matrix* A, d_matrix* B, d_matrix* R, double a /*= 1*/, double c /*= 0*/, d_matrixTrans ta /*= NoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == Trans) { std::swap(m, n); };

	if (globalUseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		for (int i = 0; i <= R->col; i++)
			cublasDgemv(handle, ta1, m, n, &a, A->data, A->row, B->data, 1, &c, R->getDataPointer(0, i), 1);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		for (int i = 0; i <= R->col; i++)
			cblas_dgemv(CblasColMajor, ta1, m, n, a, A->data, A->row, B->data, 1, c, R->getDataPointer(0, i), 1);
	}
}

void d_matrix::hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R)
{
	if (globalUseCuda)
	{
		cuda_hadamardProduct(A->data, B->data, R->data, R->max_script);
	}
	else
	{
#pragma loop(hint_parallel(8))
		for (int i = 0; i < R->max_script; i++)
		{
			R->data[i] = A->data[i] * B->data[i];
		}
	}
	/*
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
	*/
}

void d_matrix::minus(d_matrix* A, d_matrix* B, d_matrix* R)
{
	if (globalUseCuda)
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


double* d_matrix::mallocData(int size)
{
	if (UseCuda)
	{
		double* d = nullptr;
		if (cudaMalloc((void **)&d, size * sizeof(double)) == cudaSuccess)
		{
			dataIsWhere = DataInDevice;
		}
		return d;
	}
	else
	{
		return new double[size];
	}
}

void d_matrix::freeData()
{
	if (!data)
		return;
	if (UseCuda)
	{
		cudaFree(data);
		return;
	}
	else
	{
		delete data;
	}
	data = nullptr;
}

double* d_matrix::malloc_getDataFromDevice()
{
	if (UseCuda)
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
	if (UseCuda)
	{
		delete temp;
	}
}

double* d_matrix::mallocDataForDevice()
{
	if (UseCuda)
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
	if (UseCuda)
	{
		cudaMemcpy(data, temp, sizeof(double)*max_script, cudaMemcpyHostToDevice);
		delete temp;
	}
}

void d_matrix::activeFunction(d_matrix* A, d_matrix* R, ActiveFunctionMode afm)
{
	switch (afm)
	{
	case Sigmoid:
		if (globalUseCuda)
		{
			cuda_sigmoid(A->data, R->data, R->max_script);
		}
		else
		{
			MyMath::sigmoid_v(A->data, R->data, R->max_script);
		}		
		break;
	case Softmax:
		if (globalUseCuda)
		{
			cuda_exp(A->data, R->data, R->max_script);
		}
		else
		{
			MyMath::exp_v(A->data, R->data, R->max_script);
		}	
		for (int i = 0; i < R->col; i++)
		{
			double sum = R->sumColAbs(i);
			if (sum == 0) continue;
			R->colMultiply(1 / sum, i);
		}
		break;
	case Tanh:
		if (globalUseCuda)
		{

		}
		else
		{
			MyMath::tanh_v(A->data, R->data, R->max_script);
		}
		break;
	case Findmax:
		if (globalUseCuda)
		{

		}
		else
		{
			if (R->max_script <= 0) return;
			auto temp = new double[R->max_script];
			memset(temp, 0, sizeof(double)*R->max_script);
			std::swap(R->data, temp);
			delete temp;
			for (int i_group = 0; i_group < R->col; i_group++)
			{
				int index = A->indexColMaxAbs(i_group);
				R->getData(index, i_group) = 1;
			}
		}
		break;
	}
}

void d_matrix::dactiveFunction(d_matrix* A, d_matrix* R, ActiveFunctionMode afm)
{
	switch (afm)
	{
	case Sigmoid:
		if (globalUseCuda)
		{
			cuda_dsigmoid(A->data, R->data, R->max_script);
		}
		else
		{
			MyMath::dsigmoid_v(A->data, R->data, R->max_script);
		}
		break;
	case Softmax:
		//softmax一般是最后一层，可能无用
		if (globalUseCuda)
		{
			cuda_exp(A->data, R->data, R->max_script);
		}
		else
		{
			MyMath::exp_v(A->data, R->data, R->max_script);
		}
		break;
	case Tanh:
		if (globalUseCuda)
		{

		}
		else
		{
			MyMath::dtanh_v(A->data, R->data, R->max_script);
		}
		break;
	case Findmax:
		if (globalUseCuda)
		{

		}
		else
		{

		}
		break;
	}
}

