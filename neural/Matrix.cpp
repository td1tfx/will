#include "Matrix.h"

MatrixCudaType Matrix::globalUseCuda = mc_NoCuda;
bool Matrix::inited = false;

cublasHandle_t Matrix::cublasHandle;
cudnnHandle_t Matrix::cudnnHandle;
cudnnTensorDescriptor_t Matrix::td;
cudnnActivationDescriptor_t Matrix::ad;
cudnnOpTensorDescriptor_t Matrix::od;
cudnnPoolingDescriptor_t Matrix::pd;
cudnnConvolutionDescriptor_t Matrix::cd;

using namespace MyMath;

//构造函数
Matrix::Matrix(int m, int n, MatrixDataType tryInside, MatrixCudaType tryCuda)
{
	insideData = tryInside;
	UseCuda = (tryCuda == mc_UseCuda) && (globalUseCuda == mc_UseCuda) ? mc_UseCuda : mc_NoCuda;

	row = m;
	col = n;
	W = n;
	H = m;
	C = 1;
	N = 1;
	max_script = row*col;
	if (insideData == md_Inside)
	{
		data = mallocData(max_script);
		data_size = max_script;
	}
	if (UseCuda == mc_UseCuda)
	{
		cudnnCreateTensorDescriptor(&tensorDes);
		setTensorDes(tensorDes, 1, 1, n, m);
	}
}

//以4阶张量构造
Matrix::Matrix(int w, int h, int c, int n, MatrixDataType tryInside /*= md_Inside*/, MatrixCudaType tryCuda /*= mc_UseCuda*/)
:Matrix(w*h*c, n, tryInside, tryCuda)
{
	W = w;
	H = h;
	C = c;
	N = n;
	if (UseCuda == mc_UseCuda)
	{
		setTensorDes(tensorDes, n, c, h, w);
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
		setTensorDes(tensorDes, 1, 1, n, m);
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


//重设数据指针，比较危险，不推荐
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
	globalUseCuda = mc_NoCuda;
	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUBLAS initialization error\n");
		return;
	}
	if (cudnnCreate(&cudnnHandle) != CUDNN_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUDNN initialization error\n");
		return;
	}
	dev = findCudaDevice(0, nullptr);
	if (dev >= 0)
	{
		globalUseCuda = mc_UseCuda;
	}
	cudnnCreateTensorDescriptor(&td);
	cudnnCreateActivationDescriptor(&ad);
	cudnnCreateOpTensorDescriptor(&od);
	cudnnCreatePoolingDescriptor(&pd);
	cudnnCreateConvolutionDescriptor(&cd);
#endif	
}

void Matrix::destroyCuda()
{
#ifdef _USE_CUDA
	cudnnDestroyTensorDescriptor(td);
	cudnnDestroyActivationDescriptor(ad);
	cudnnDestroyOpTensorDescriptor(od);
	cudnnDestroyPoolingDescriptor(pd);
	cudnnDestroyConvolutionDescriptor(cd);

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
			double v = temp[xy2i(i, j)];
			if (std::abs(v) > 1e10)
				fprintf(fout, "%14.11e ", v);
			else
				fprintf(fout, "%14.11lf ", v);
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

//将矩阵当做向量，按照内存中的顺序依次输出
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

//将矩阵当做向量，按照内存中的顺序依次载入
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

//将外界的值复制到矩阵，参数指针必须指向Host内存！
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

//将矩阵的值复制到外界，参数指针必须指向Host内存！
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

//一列中最大值的序号
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

//一列的绝对值和
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

//点乘，即所有元素平方和
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

//以同一个值初始化矩阵
void Matrix::initData(double v)
{
	if (UseCuda == mc_UseCuda)
	{
		setTensorDes(td, 1, 1, col, row);
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


//随机数初始化矩阵，注意这个函数调用次数很少
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

//用连续整数初始化，仅用于测试
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

//数乘
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

//选择一列数乘
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

//将显存中的数据转移到内存
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

//将一个外部数据矩阵的指针指向其他位置
void Matrix::shareData(Matrix* A, int m, int n)
{
	if (insideData == md_Outside && UseCuda == A->UseCuda)
		this->data = A->getDataPointer(m, n);
}

//矩阵乘，R = aAB+cR
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

//矩阵乘以向量，R = aAB+cR
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

//没什么用，废弃
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

//矩阵元素乘
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

//矩阵减
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

//池化
void Matrix::poolingForward(ResampleType re, Matrix* X, Matrix* Y, 
	int window_w, int window_h, int stride_w, int stride_h, int** maxPos /*= nullptr*/)
{
	if (globalUseCuda == mc_UseCuda)
	{
		double a = 1, b = 0;
		auto pm = re == re_Max ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		cudnnSetPooling2dDescriptor(pd, pm, CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
		cudnnPoolingForward(cudnnHandle, pd, &a, X->tensorDes, X->data, &b, Y->tensorDes, Y->data);
	}
	else
	{
		for (int p = 0; p < Y->N*Y->C; p++)
		{
			for (int i_Y = 0; i_Y < Y->W; i_Y++)
			{
				for (int j_Y = 0; j_Y < Y->H; j_Y++)
				{
					double v = 0;
					//if (re == re_Average)v = 0;
					if (re == re_Max) v = -DBL_MAX;
					for (int i_X = i_Y*stride_w; i_X < std::min(X->W, i_Y*stride_w + window_w); i_X++)
					{
						for (int j_X = j_Y*stride_h; j_X < std::min(X->H, j_Y*stride_h + window_h); j_X++)
						{
							if (re == re_Average)
							{
								v += X->getData(i_X, j_X, p);
							}
							else if (re == re_Max)
							{
								auto x = X->getData(i_X, j_X, p);
								if (x > v) 
								{
									v = x;
									(*maxPos)[i_Y + j_Y*Y->W + p*Y->H*Y->W] = i_X + j_X*X->W + p*X->H*X->W;
								}
							}								
						}
					}
					if (re == re_Average) v /= window_w*window_h;
					Y->getData(i_Y, j_Y, p) = v;
				}
			}
		}
	}
}

void Matrix::poolingBackward(ResampleType re, Matrix* Y, Matrix* DY, Matrix* X, Matrix* DX, 
	int window_w, int window_h, int stride_w, int stride_h, int* maxPos /*= nullptr*/)
{
	if (globalUseCuda == mc_UseCuda)
	{
		//这个怎么看都快不了
		double a = 1, b = 0;
		auto pm = re == re_Max ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		cudnnSetPooling2dDescriptor(pd, pm, CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
		cudnnPoolingBackward(cudnnHandle, pd, &a, Y->tensorDes, Y->data, DY->tensorDes, DY->data, X->tensorDes, X->data, &b, DX->tensorDes, DX->data);
	}
	else
	{
		if (re == re_Average)
		{
			for (int p = 0; p < DY->N*DY->C; p++)
			{
				for (int i_DY = 0; i_DY < DY->W; i_DY++)
				{
					for (int j_DY = 0; j_DY < DY->H; j_DY++)
					{
						double v = DY->getData(i_DY, j_DY, p) / window_w / window_h;
						for (int i_DX = i_DY*stride_w; i_DX < std::min(DX->W, i_DY*stride_w + window_w); i_DX++)
						{
							for (int j_DX = j_DY*stride_h; j_DX < std::min(DX->H, j_DY*stride_h + window_h); j_DX++)
							{
								DX->getData(i_DX, j_DX, p) = v;
							}
						}
					}
				}
			}
		}
		else if (re == re_Max || maxPos)
		{
			//这样速度会快一点
			DX->initData(0);
			for (int i = 0; i < DY->getDataCount(); i++)
			{
				DX->getData(maxPos[i]) = DY->getData(i);
			}
		}
	}
}

void Matrix::convolution(Matrix* A, Matrix* conv_kernel, Matrix* R, int m_subA, int n_subA, int m_subR, int n_subR, int countPerGroup)
{
	if (globalUseCuda == mc_UseCuda)
	{
	}
	else
	{
		for (int p = 0; p < R->N*R->C; p++)
		{
			for (int i_R = 0; i_R < R->W; i_R++)
			{
				for (int j_R = 0; j_R < R->H; j_R++)
				{
					double v = 0;
					for (int i_A = i_R; i_A < std::min(A->W, i_R + conv_kernel->row); i_A++)
					{
						for (int j_A = j_R; j_A < std::min(A->H, j_R + conv_kernel->col); j_A++)
						{
							v += A->getData(i_A, j_A, p) * conv_kernel->getData(i_A - i_R, j_A - j_R);
						}
					}
					R->getData(i_R, j_R, p) = v;
				}
			}
		}
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

void Matrix::setTensorDes(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w)
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

void Matrix::activeForward(ActiveFunctionType af, Matrix* X, Matrix* Y)
{
	double a = 1, b = 0;
	switch (af)
	{
	case af_Sigmoid:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			cudnnActivationForward(cudnnHandle, ad, &a, X->tensorDes, X->data, &b, Y->tensorDes, Y->data);
		}
		else
		{
			MyMath::sigmoid_v(X->data, Y->data, Y->max_script);
		}
		break;
	case af_Linear:
		cpyData(Y, X);
		break;
	case af_Softmax:
		if (globalUseCuda == mc_UseCuda)
		{
			setTensorDes(td, X->col, 1, 1, X->row);
			cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
				&a, td, X->data, &b, td, Y->data);
		}
		else
		{
			MyMath::exp_v(X->data, Y->data, Y->max_script);
			for (int i = 0; i < Y->col; i++)
			{
				double sum = Y->sumColAbs(i);
				if (sum == 0) continue;
				Y->colMultiply(1 / sum, i);
			}
		}
		break;
	case af_Tanh:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_TANH);
			cudnnActivationForward(cudnnHandle, ad, &a, X->tensorDes, X->data, &b, Y->tensorDes, Y->data);
		}
		else
		{
			MyMath::tanh_v(X->data, Y->data, Y->max_script);
		}
		break;
	case af_Findmax:
		if (globalUseCuda == mc_UseCuda)
		{

		}
		else
		{
			if (Y->max_script <= 0) return;
			auto temp = new double[Y->max_script];
			memset(temp, 0, sizeof(double)*Y->max_script);
			std::swap(Y->data, temp);
			delete temp;
			for (int i_group = 0; i_group < Y->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				Y->getData(index, i_group) = 1;
			}
		}
		break;
	case af_Softplus:
		if (globalUseCuda == mc_UseCuda)
		{

		}
		else
		{
			MyMath::softplus_v(X->data, Y->data, Y->max_script);
		}
		break;
	case af_ReLU:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_RELU);
			cudnnActivationForward(cudnnHandle, ad, &a, X->tensorDes, X->data, &b, Y->tensorDes, Y->data);
		}
		else
		{
			MyMath::relu_v(X->data, Y->data, Y->max_script);
		}
		break;
	}
}

void Matrix::activeBackward(ActiveFunctionType af, Matrix* Y, Matrix* X, Matrix* DX)
{
	double a = 1, b = 0;
	switch (af)
	{
	case af_Sigmoid:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			cudnnActivationBackward(cudnnHandle, ad, &a, Y->tensorDes, Y->data, DX->tensorDes, DX->data,
				X->tensorDes, X->data, &b, DX->tensorDes, DX->data);
		}
		else
		{
			MyMath::sigmoid_vb(X->data, DX->data, DX->max_script);
		}
		break;
	case af_Linear:
		DX->initData(1);
		break;
	case af_Softmax:
		//softmax一般是最后一层，可能无用
		if (globalUseCuda == mc_UseCuda)
		{
			setTensorDes(td, X->col, 1, 1, X->row);
			//TODO: wei wan cheng
			//cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
			//	&alpha, td, A->data, &beta, td, R->data);
		}
		else
		{
			MyMath::exp_v(X->data, DX->data, DX->max_script);
		}
		break;
	case af_Tanh:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_TANH);
			cudnnActivationBackward(cudnnHandle, ad, &a, Y->tensorDes, Y->data, DX->tensorDes, DX->data,
				X->tensorDes, X->data, &b, DX->tensorDes, DX->data);
		}
		else
		{
			MyMath::tanh_vb(X->data, DX->data, DX->max_script);
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
			cudnnActivationForward(cudnnHandle, ad, &a, X->tensorDes, X->data, &b, DX->tensorDes, DX->data);
		}
		else
		{
			MyMath::softplus_vb(X->data, DX->data, DX->max_script);
		}
		break;
	case af_ReLU:
		if (globalUseCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_RELU);
			cudnnActivationBackward(cudnnHandle, ad, &a, Y->tensorDes, Y->data, DX->tensorDes, DX->data,
				X->tensorDes, X->data, &b, DX->tensorDes, DX->data);
		}
		else
		{
			MyMath::relu_vb(X->data, DX->data, DX->max_script);
		}
		break;
	}
}

