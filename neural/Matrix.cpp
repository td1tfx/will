#include "Matrix.h"

MatrixCudaType Matrix::globalUseCuda = mc_NoCuda;
bool Matrix::inited = false;

cublasHandle_t Matrix::cublasHandle;
cudnnHandle_t Matrix::cudnnHandle;
cudnnTensorDescriptor_t Matrix::td;
cudnnActivationDescriptor_t Matrix::ad;
cudnnOpTensorDescriptor_t Matrix::od;

using namespace MyMath;

Matrix::Matrix(int m, int n, MatrixDataType tryInside, MatrixCudaType tryCuda)
{
	insideData = tryInside;
	UseCuda = (tryCuda == mc_UseCuda) && (globalUseCuda == mc_UseCuda) ? mc_UseCuda : mc_NoCuda;

	row = m;
	col = n;
	max_script = row*col;
	if (insideData == md_Inside)
	{
		data = mallocData(max_script);
		data_size = max_script;
	}
	if (UseCuda == mc_UseCuda)
	{
		cudnnCreateTensorDescriptor(&tensorDes);
		setTensor(tensorDes, 1, 1, n, m);
	}
}

Matrix::Matrix(int w, int h, int c, int n, MatrixDataType tryInside /*= md_Inside*/, MatrixCudaType tryCuda /*= mc_UseCuda*/)
{
	Matrix(w*h*c, h, tryInside, tryCuda);
	if (UseCuda == mc_UseCuda)
	{
		setTensor(tensorDes, n, c, h, w);
	}
}

Matrix::~Matrix()
{
	if (insideData == md_Inside) freeData();
	if (UseCuda == mc_UseCuda) cudnnDestroyTensorDescriptor(tensorDes);
}

//返回值：-1空矩阵，未重新分配内存，1重新分配内存
int Matrix::resize(int m, int n, int force /*= 0*/)
{
	if (!this)
		return -1;
	row = m;
	col = n;
	max_script = m*n;
	if (UseCuda == mc_UseCuda)
	{
		setTensor(tensorDes, 1, 1, n, m);
	}
	//空间不够或者强制则重新分配
	if (max_script > data_size || force)
	{
		//重新申请空间
		if (insideData == md_Inside)
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
void Matrix::resetDataPointer(double* d, int d_in_cuda /*= 0*/)
{
	if (UseCuda == mc_UseCuda)
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

void Matrix::initCuda()
{
	if (inited) { return; }
	inited = true;
#ifdef _USE_CUDA
	int dev = -1;
	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUBLAS initialization error\n");
	}

	if (cudnnCreate(&cudnnHandle) != CUDNN_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUDNN initialization error\n");
	}

	dev = findCudaDevice(0, nullptr);
	globalUseCuda = (dev >= 0) ? mc_UseCuda : mc_NoCuda;
	cudnnCreateTensorDescriptor(&td);
	cudnnCreateActivationDescriptor(&ad);
	cudnnCreateOpTensorDescriptor(&od);
#endif	
}

void Matrix::destroyCuda()
{
#ifdef _USE_CUDA
	cudnnDestroyTensorDescriptor(td);
	cudnnDestroyActivationDescriptor(ad);
	cudnnDestroyOpTensorDescriptor(od);

	cublasDestroy(cublasHandle);
	cudnnDestroy(cudnnHandle);
#endif
}

void Matrix::print(FILE* fout)
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

int Matrix::load(double* v, int n)
{
	auto temp = mallocDataForDevice();
	int k = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp[xy2i(i, j)] = v[k++];
			if (k >= n) break;
		}
	}
	set_freeDataToDevice(temp);
	return k;
}

void Matrix::printAsVector(FILE* fout /*= stdout*/)
{
	auto temp = malloc_getDataFromDevice();
	for (int i = 0; i < max_script; i++)
	{
		fprintf(fout, "%14.11lf ", temp[i]);
	}
	fprintf(fout, "\n");
	freeDataForDevice(temp);
}

int Matrix::loadAsVector(double* v, int n)
{
	auto temp = mallocDataForDevice();
	int k = 0;
	for (int i = 0; i < row; i++)
	{
		temp[i] = v[k++];
		if (k >= n) break;
	}
	set_freeDataToDevice(temp);
	return k;
}

//参数指针必须指向Host内存！
void Matrix::memcpyDataIn(double* src, int size)
{
	if (UseCuda == mc_UseCuda)
	{
		cudaMemcpy(data, src, int(sizeof(double)*std::min(size, max_script)), cudaMemcpyHostToDevice);
	}
	else
	{
		memcpy(data, src, int(sizeof(double)*std::min(size, max_script)));
	}
}

//参数指针必须指向Host内存！
void Matrix::memcpyDataOut(double* dst, int size)
{
	if (UseCuda == mc_UseCuda)
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
void Matrix::expand()
{
	if (UseCuda == mc_UseCuda)
	{
		for (int i = 1; i < col; i *= 2)
		{
			cudaMemcpy(getDataPointer(0, i), getDataPointer(0, 0),
				sizeof(double)*row*std::min(i, col - i), cudaMemcpyDeviceToDevice);
		}
	}
	else
	{
		//#pragma loop(hint_parallel(8))
		for (int i = 1; i < col; i *= 2)
		{
			memcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(double)*row*std::min(i, col - i));
		}
	}
}

int Matrix::indexColMaxAbs(int c)
{
	if (UseCuda == mc_UseCuda)
	{
		int r;
		cublasIdamax(cublasHandle, row, getDataPointer(0, c), 1, &r);
		return r - 1;
	}
	else
	{
		return cblas_idamax(row, getDataPointer(0, c), 1);
	}
}

double Matrix::sumColAbs(int c)
{
	if (UseCuda == mc_UseCuda)
	{
		double r;
		cublasDasum(cublasHandle, row, getDataPointer(0, c), 1, &r);
		return r;
	}
	else
	{
		return cblas_dasum(row, getDataPointer(0, c), 1);
	}
}

double Matrix::ddot()
{
	if (UseCuda == mc_UseCuda)
	{
		double r;
		cublasDdot(cublasHandle, max_script, data, 1, data, 1, &r);
		return r;
	}
	else
	{
		return cblas_ddot(max_script, data, 1, data, 1);
	}
}

void Matrix::initData(double v)
{
	if (UseCuda == mc_UseCuda)
	{
		setTensor(td, 1, 1, col, row);
		cudnnSetTensor(cudnnHandle, td, data, &v);
	}
	else
	{
#pragma loop(hint_parallel(8))
		for (int i = 0; i < max_script; i++)
		{
			data[i] = v;
		}
	}
}


//注意这个函数调用次数很少
void Matrix::initRandom()
{
	auto temp = mallocDataForDevice();
	//#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		temp[i] = 2.0 * rand() / RAND_MAX - 1;
	}
	set_freeDataToDevice(temp);
}

//用连续整数初始化，用于测试
void Matrix::initInt()
{
	auto temp = mallocDataForDevice();
	//#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		temp[i] = i;
	}
	set_freeDataToDevice(temp);
}

void Matrix::multiply(double v)
{
	if (UseCuda == mc_UseCuda)
	{
		cublasDscal(cublasHandle, row, &v, data, 1);
	}
	else
	{
		cblas_dscal(max_script, v, data, 1);
	}
}

void Matrix::colMultiply(double v, int c)
{
	if (UseCuda == mc_UseCuda)
	{
		cublasDscal(cublasHandle, row, &v, getDataPointer(0, c), 1);
	}
	else
	{
		cblas_dscal(row, v, getDataPointer(0, c), 1);
	}
}


//复制数据，只处理较少的
void Matrix::cpyData(Matrix* dst, Matrix* src)
{
	if (globalUseCuda == mc_UseCuda)
	{
		cudaMemcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col), cudaMemcpyDeviceToDevice);
	}
	else
	{
		memcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col));
	}
}

void Matrix::tryUploadToCuda()
{
	if (globalUseCuda == mc_UseCuda)
	{
		if (UseCuda == mc_NoCuda)
		{
			UseCuda = mc_UseCuda;
			auto temp = mallocData(data_size);
			if (temp)
			{
				std::swap(temp, data);
				set_freeDataToDevice(temp);
			}
			else
			{
				UseCuda = mc_NoCuda;
			}
		}
	}
}

void Matrix::tryDownloadFromCuda()
{
	if (UseCuda == mc_UseCuda)
	{
		auto temp = malloc_getDataFromDevice();
		if (temp)
		{
			std::swap(temp, data);
			cudaFree(temp);
		}
		UseCuda = mc_NoCuda;
	}
}

void Matrix::shareData(Matrix* A, int m, int n)
{
	if (insideData == md_Outside && UseCuda == A->UseCuda)
		this->data = A->getDataPointer(m, n);
}

void Matrix::product(Matrix* A, Matrix* B, Matrix* R,
	double a /*= 1*/, double c /*= 0*/, MatrixTransType ta /*= NoTrans*/, MatrixTransType tb /*= NoTrans*/)
{
	int m = R->row;
	int n = R->col;
	int lda = A->row;
	int k = A->col;
	int ldb = B->row;
	if (ta == mt_Trans) { k = A->row; }
	if (globalUseCuda == mc_UseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		auto tb1 = get_cublas_trans(tb);
		cublasDgemm(cublasHandle, ta1, tb1, m, n, k, &a, A->data, lda, B->data, ldb, &c, R->data, m);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		auto tb1 = get_cblas_trans(tb);
		cblas_dgemm(CblasColMajor, ta1, tb1, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
	}
}

void Matrix::productVector(Matrix* A, Matrix* B, Matrix* R, double a /*= 1*/, double c /*= 0*/, MatrixTransType ta /*= NoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == mt_Trans) { std::swap(m, n); };

	if (globalUseCuda == mc_UseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		cublasDgemv(cublasHandle, ta1, m, n, &a, A->data, A->row, B->data, 1, &c, R->data, 1);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		cblas_dgemv(CblasColMajor, ta1, m, n, a, A->data, A->row, B->data, 1, c, R->data, 1);
	}
}

void Matrix::productVector2(Matrix* A, Matrix* B, Matrix* R, double a /*= 1*/, double c /*= 0*/, MatrixTransType ta /*= NoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == mt_Trans) { std::swap(m, n); };

	if (globalUseCuda == mc_UseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		for (int i = 0; i <= R->col; i++)
			cublasDgemv(cublasHandle, ta1, m, n, &a, A->data, A->row, B->data, 1, &c, R->getDataPointer(0, i), 1);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		for (int i = 0; i <= R->col; i++)
			cblas_dgemv(CblasColMajor, ta1, m, n, a, A->data, A->row, B->data, 1, c, R->getDataPointer(0, i), 1);
	}
}


void Matrix::hadamardProduct(Matrix* A, Matrix* B, Matrix* R)
{
	if (globalUseCuda == mc_UseCuda)
	{
		double a1 = 1, a2 = 1, b = 0;
		cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_DOUBLE, CUDNN_NOT_PROPAGATE_NAN);
		cudnnOpTensor(cudnnHandle, od, &a1, A->tensorDes, A->data, &a2, B->tensorDes, B->data, &b, R->tensorDes, R->data);
	}
	else
	{
#pragma loop(hint_parallel(8))
		for (int i = 0; i < R->max_script; i++)
		{
			R->data[i] = A->data[i] * B->data[i];
		}
	}
}


void Matrix::minus(Matrix* A, Matrix* B, Matrix* R)
{
	if (globalUseCuda == mc_UseCuda)
	{
		double a = -1;
		cublasDcopy(cublasHandle, R->max_script, A->data, 1, R->data, 1);
		cublasDaxpy(cublasHandle, R->max_script, &a, B->data, 1, R->data, 1);

		//double a1 = 1, a2 = -1, b = 0;
		//setTensor(td, 1, 1, R->col, R->row);
		//cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_DOUBLE, CUDNN_NOT_PROPAGATE_NAN);
		//cudnnOpTensor(cudnnHandle, od, &a1, td, A->data, &a2, td, B->data, &b, td, R->data);
	}
	else
	{
		cblas_dcopy(R->max_script, A->data, 1, R->data, 1);
		cblas_daxpy(R->max_script, -1, B->data, 1, R->data, 1);
		// #pragma loop(hint_parallel(8))
		// 	for (int i = 0; i < R->max_script; i++)
		// 	{
		// 		R->data[i] = A->data[i] - B->data[i];
		// 	}
	}
}


void Matrix::resample(Matrix* A, Matrix* R, ResampleType re, int** maxPos, int basePos)
{
	int scalem = (A->row + R->row - 1) / R->row;
	int scalen = (A->col + R->col - 1) / R->col;
	if (globalUseCuda == mc_UseCuda)
	{
	}
	else
	{
		for (int i1 = 0; i1 < A->row; i1 += scalem)
		{
			for (int j1 = 0; j1 < A->col; j1 += scalen)
			{
				double v = -1;
				for (int i2 = i1; i2 < std::min(i1 + scalem, A->row); i2++)
				{
					for (int j2 = j1; j2 < std::min(j1 + scalen, A->col); j2++)
					{
						double d = A->getData(i2, j2);
						if (re == 0)
						{
							if (d > v)
							{
								v = d;
								if (maxPos)
								{
									(*maxPos)[R->xy2i(i1 / scalem, j1 / scalen)] = A->xy2i(i2, j2) + basePos;
								}
							}
						}
						else
						{
							v += d;
						}
					}
				}
				if (re == 1)
				{
					v /= scalem*scalen;
				}
				R->getData(i1 / scalem, j1 / scalen) = v;
			}
		}
	}
}

void Matrix::resample_colasImage(Matrix* A, Matrix* R, int m_subA, int n_subA, int m_subR, int n_subR,
	int countPerGroup, ResampleType re, int** maxPos /*= nullptr*/)
{
	auto subA = new Matrix(m_subA, n_subA, md_Outside, mc_UseCuda);
	auto subR = new Matrix(m_subR, n_subR, md_Outside, mc_UseCuda);
	for (int i = 0; i < countPerGroup; i++)
	{
		for (int j = 0; j < A->col; j++)
		{
			subA->shareData(A, i*subA->max_script, j);
			subR->shareData(R, i*subR->max_script, j);
			resample(subA, subR, re, maxPos ? maxPos + i*subA->max_script : nullptr, i*subA->max_script);
		}
	}
	delete subA;
	delete subR;
}

void Matrix::convolution(Matrix* A, Matrix* conv_kernel, Matrix* R)
{
	if (globalUseCuda == mc_UseCuda)
	{
	}
	else
	{
		for (int i1 = 0; i1 < A->row + 1 - conv_kernel->row; i1++)
		{
			for (int j1 = 0; j1 < A->col + 1 - conv_kernel->col; j1++)
			{
				double v = 0;
				for (int i2 = 0; i2 < conv_kernel->row; i2++)
				{
					for (int j2 = 0; j2 < conv_kernel->col; j2++)
					{
						double d = A->getData(i1 + i2, j1 + j2)*conv_kernel->getData(i2, j2);
						v += d;
					}
				}
				R->getData(i1, j1) = v;
			}
		}
	}
}

void Matrix::convolution_colasImage(Matrix* A, Matrix* conv_kernel, Matrix* R, int m_subA, int n_subA, int m_subR, int n_subR, int countPerGroup)
{
	auto subA = new Matrix(m_subA, n_subA, md_Outside, mc_UseCuda);
	auto subR = new Matrix(m_subR, n_subR, md_Outside, mc_UseCuda);
	for (int i = 0; i < countPerGroup; i++)
	{
		for (int j = 0; j < A->col; j++)
		{
			subA->shareData(A, i*subA->max_script, j);
			subR->shareData(R, i*subR->max_script, j);
			convolution(subA, conv_kernel, subR);
		}
	}
	delete subA;
	delete subR;
}

double* Matrix::mallocData(int size)
{
	if (UseCuda == mc_UseCuda)
	{
		double* d = nullptr;
		if (cudaMalloc((void **)&d, size * sizeof(double)) == cudaSuccess)
		{
			//dataIsWhere = DataInDevice;
		}
		return d;
	}
	else
	{
		return new double[size];
	}
}

void Matrix::freeData()
{
	if (!data)
		return;
	if (UseCuda == mc_UseCuda)
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

double* Matrix::malloc_getDataFromDevice()
{
	if (UseCuda == mc_UseCuda)
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

void Matrix::freeDataForDevice(double* temp)
{
	if (UseCuda == mc_UseCuda)
	{
		delete temp;
	}
}

double* Matrix::mallocDataForDevice()
{
	if (UseCuda == mc_UseCuda)
	{
		return new double[max_script];
	}
	else
	{
		return data;
	}
}

void Matrix::set_freeDataToDevice(double* temp)
{
	if (UseCuda == mc_UseCuda)
	{
		cudaMemcpy(data, temp, sizeof(double)*max_script, cudaMemcpyHostToDevice);
		delete temp;
	}
}

//这里应该有优化的办法，再说
void Matrix::selectFunction(MatrixCudaType useCuda, double* x, double* y, int size,
	std::function<int(double*, double*, int)> f1, std::function<int(double*, double*, int)> f2)
{
	if (useCuda == mc_UseCuda)
	{
		f1(x, y, size);
	}
	else
	{
		f2(x, y, size);
	}
}

void Matrix::setTensor(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w)
{
	cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w);
}

void Matrix::setActive(cudnnActivationMode_t am)
{
	cudnnSetActivationDescriptor(ad, am, CUDNN_NOT_PROPAGATE_NAN, 1);
}

void Matrix::setActiveParameter(cudnnActivationMode_t am, int n, int c, int h, int w)
{
	cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w);
	cudnnSetActivationDescriptor(ad, am, CUDNN_NOT_PROPAGATE_NAN, 1);
}

void Matrix::activeForward(ActiveFunctionType af, Matrix* A, Matrix* R)
{
	double alpha = 1, beta = 0;
	switch (af)
	{
	case af_Sigmoid:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			cudnnActivationForward(cudnnHandle, ad, &alpha, A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::sigmoid_v(A->data, R->data, R->max_script);
		}
		break;
	case af_Linear:
		cpyData(R, A);
		break;
	case af_Softmax:
		if (globalUseCuda == mc_UseCuda)
		{
			setTensor(td, A->col, 1, 1, A->row);
			cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
				&alpha, td, A->data, &beta, td, R->data);
		}
		else
		{
			MyMath::exp_v(A->data, R->data, R->max_script);
			for (int i = 0; i < R->col; i++)
			{
				double sum = R->sumColAbs(i);
				if (sum == 0) continue;
				R->colMultiply(1 / sum, i);
			}
		}
		break;
	case af_Tanh:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_TANH);
			cudnnActivationForward(cudnnHandle, ad, &alpha, A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::tanh_v(A->data, R->data, R->max_script);
		}
		break;
	case af_Findmax:
		if (globalUseCuda == mc_UseCuda)
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
	case af_Softplus:
		if (globalUseCuda == mc_UseCuda)
		{

		}
		else
		{
			MyMath::softplus_v(A->data, R->data, R->max_script);
		}
		break;
	case af_ReLU:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_RELU);
			cudnnActivationForward(cudnnHandle, ad, &alpha, A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::relu_v(A->data, R->data, R->max_script);
		}
		break;
	}
}

void Matrix::activeBackward(ActiveFunctionType af, Matrix* A, Matrix* B, Matrix* R)
{
	double alpha = 1, beta = 0;
	switch (af)
	{
	case af_Sigmoid:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			//这里没有用到y矩阵
			cudnnActivationBackward(cudnnHandle, ad, &alpha, B->tensorDes, B->data, R->tensorDes, R->data, 
				A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::sigmoid_vb(A->data, R->data, R->max_script);
		}
		break;
	case af_Linear:
		R->initData(1);
		break;
	case af_Softmax:
		//softmax一般是最后一层，可能无用
		if (globalUseCuda == mc_UseCuda)
		{
			setTensor(td, A->col, 1, 1, A->row);
			//TODO: wei wan cheng
			//cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
			//	&alpha, td, A->data, &beta, td, R->data);
		}
		else
		{
			MyMath::exp_v(A->data, R->data, R->max_script);
		}
		break;
	case af_Tanh:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_TANH);
			cudnnActivationBackward(cudnnHandle, ad, &alpha, B->tensorDes, B->data, R->tensorDes, R->data,
				A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::tanh_vb(A->data, R->data, R->max_script);
		}
		break;
	case af_Findmax:
		if (globalUseCuda == mc_UseCuda)
		{

		}
		else
		{

		}
		break;
	case af_Softplus:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			cudnnActivationForward(cudnnHandle, ad, &alpha, A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::softplus_vb(A->data, R->data, R->max_script);
		}
		break;
	case af_ReLU:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_RELU);
			cudnnActivationBackward(cudnnHandle, ad, &alpha, B->tensorDes, B->data, R->tensorDes, R->data,
				A->tensorDes, A->data, &beta, R->tensorDes, R->data);
		}
		else
		{
			MyMath::relu_vb(A->data, R->data, R->max_script);
		}
		break;
	}
}

