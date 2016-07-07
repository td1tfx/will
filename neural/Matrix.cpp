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
cudnnFilterDescriptor_t Matrix::fd;

using namespace MyMath;

//普通二维矩阵构造函数
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
	if (globalUseCuda == mc_UseCuda)
	{
		cudnnCreateTensorDescriptor(&tensorDes);
	}
	setTensorDes(tensorDes, 1, 1, n, m);
}

//4阶张量形式构造函数，用于池化和卷积
Matrix::Matrix(int w, int h, int c, int n, MatrixDataType tryInside /*= md_Inside*/, MatrixCudaType tryCuda /*= mc_UseCuda*/)
	:Matrix(w*h*c, n, tryInside, tryCuda)
{
	W = w;
	H = h;
	C = c;
	N = n;
	setTensorDes(tensorDes, n, c, h, w);
}

Matrix::~Matrix()
{
	if (insideData == md_Inside) freeData();
	if (tensorDes) cudnnDestroyTensorDescriptor(tensorDes);
}

//返回值：-1空矩阵，未重新分配内存，1重新分配内存
int Matrix::resize(int m, int n, int force /*= 0*/)
{
	if (!this)
		return -1;
	row = m;
	col = n;
	max_script = m*n;
	setTensorDes(tensorDes, 1, 1, n, m);
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
void Matrix::resetDataPointer(real* d, int d_in_cuda /*= 0*/)
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

//在matrix中初始化Cuda可能不是很好，暂时没想出更好的设计
void Matrix::initCuda()
{
	if (inited) { return; }
	inited = true;
#ifdef _USE_CUDA
	int dev = -1;
	globalUseCuda = mc_NoCuda;
	dev = findCudaDevice(0, nullptr);
	if (dev >= 0)
	{
		globalUseCuda = mc_UseCuda;
	}
	else
	{
		fprintf(stderr, "Cannot find CUDA device!\n");
		return;
	}

	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUBLAS initialization error!\n");
		return;
	}
	if (cudnnCreate(&cudnnHandle) != CUDNN_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUDNN initialization error!\n");
		return;
	}

	cudnnCreateTensorDescriptor(&td);
	cudnnCreateActivationDescriptor(&ad);
	cudnnCreateOpTensorDescriptor(&od);
	cudnnCreatePoolingDescriptor(&pd);
	cudnnCreateConvolutionDescriptor(&cd);
	cudnnCreateFilterDescriptor(&fd);
#endif	
}

void Matrix::destroyCuda()
{
	inited = false;
#ifdef _USE_CUDA
	cudnnDestroyTensorDescriptor(td);
	cudnnDestroyActivationDescriptor(ad);
	cudnnDestroyOpTensorDescriptor(od);
	cudnnDestroyPoolingDescriptor(pd);
	cudnnDestroyConvolutionDescriptor(cd);
	cudnnDestroyFilterDescriptor(fd);

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
			real v = temp[xy2i(i, j)];
			if (std::abs(v) > 1e10)
				fprintf(fout, "%14.11e ", v);
			else
				fprintf(fout, "%14.11f ", v);
		}
		fprintf(fout, "\n");
	}
	freeDataForDevice(temp);
}

int Matrix::load(real* v, int n)
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
		fprintf(fout, "%14.11f ", temp[i]);
	}
	fprintf(fout, "\n");
	freeDataForDevice(temp);
}

//将矩阵当做向量，按照内存中的顺序依次载入
int Matrix::loadAsVector(real* v, int n)
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
void Matrix::memcpyDataIn(real* src, int size)
{
	if (UseCuda == mc_UseCuda)
	{
		cudaMemcpy(data, src, int(sizeof(real)*std::min(size, max_script)), cudaMemcpyHostToDevice);
	}
	else
	{
		memcpy(data, src, int(sizeof(real)*std::min(size, max_script)));
	}
}

//将矩阵的值复制到外界，参数指针必须指向Host内存！
void Matrix::memcpyDataOut(real* dst, int size)
{
	if (UseCuda == mc_UseCuda)
	{
		cudaMemcpy(dst, data, int(sizeof(real)*std::min(size, max_script)), cudaMemcpyDeviceToHost);
	}
	else
	{
		memcpy(dst, data, int(sizeof(real)*std::min(size, max_script)));
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
				sizeof(real)*row*std::min(i, col - i), cudaMemcpyDeviceToDevice);
		}
	}
	else
	{
		//#pragma loop(hint_parallel(8))
		for (int i = 1; i < col; i *= 2)
		{
			memcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(real)*row*std::min(i, col - i));
		}
	}
}

//一列中最大值的序号
int Matrix::indexColMaxAbs(int c)
{
	if (UseCuda == mc_UseCuda)
	{
		int r;
		CUBLAS_FUNC_I(amax)(cublasHandle, row, getDataPointer(0, c), 1, &r);
		return r - 1;
	}
	else
	{
		return CBLAS_FUNC_I(amax)(row, getDataPointer(0, c), 1);
	}
}

real Matrix::sumAbs()
{
	if (UseCuda == mc_UseCuda)
	{
		real r;
		CUBLAS_FUNC(asum)(cublasHandle, max_script, data, 1, &r);
		return r;
	}
	else
	{
		return CBLAS_FUNC(asum)(max_script, data, 1);
	}
}

//一列的绝对值和
real Matrix::sumColAbs(int c)
{
	if (UseCuda == mc_UseCuda)
	{
		real r;
		CUBLAS_FUNC(asum)(cublasHandle, row, getDataPointer(0, c), 1, &r);
		return r;
	}
	else
	{
		return CBLAS_FUNC(asum)(row, getDataPointer(0, c), 1);
	}
}

//点乘，即所有元素平方和
real Matrix::ddot()
{
	if (UseCuda == mc_UseCuda)
	{
		real r;
		CUBLAS_FUNC(dot)(cublasHandle, max_script, data, 1, data, 1, &r);
		return r;
	}
	else
	{
		return CBLAS_FUNC(dot)(max_script, data, 1, data, 1);
	}
}

//以同一个值初始化矩阵
void Matrix::initData(real v)
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
	Random r;
	//r.set_uniform(0, 1);
	auto temp = mallocDataForDevice();
	//#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		temp[i] = 2.0 * r.rand_uniform() - 1;
		//temp[i] = r.rand_normal();
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
void Matrix::multiply(real v)
{
	if (UseCuda == mc_UseCuda)
	{
		CUBLAS_FUNC(scal)(cublasHandle, row, &v, data, 1);
	}
	else
	{
		CBLAS_FUNC(scal)(max_script, v, data, 1);
	}
}

//选择一列数乘
void Matrix::colMultiply(real v, int c)
{
	if (UseCuda == mc_UseCuda)
	{
		CUBLAS_FUNC(scal)(cublasHandle, row, &v, getDataPointer(0, c), 1);
	}
	else
	{
		CBLAS_FUNC(scal)(row, v, getDataPointer(0, c), 1);
	}
}

//复制数据，只处理较少的
void Matrix::cpyData(Matrix* dst, Matrix* src)
{
	auto size = sizeof(real)*std::min(dst->row*dst->col, src->row*src->col);
	if (dst->UseCuda == mc_UseCuda && src->UseCuda == mc_UseCuda)
	{
		cudaMemcpy(dst->data, src->data, size, cudaMemcpyDeviceToDevice);
	}
	else if (dst->UseCuda == mc_UseCuda && src->UseCuda == mc_NoCuda)
	{
		cudaMemcpy(dst->data, src->data, size, cudaMemcpyHostToDevice);
	}
	else if (dst->UseCuda == mc_NoCuda && src->UseCuda == mc_UseCuda)
	{
		cudaMemcpy(dst->data, src->data, size, cudaMemcpyDeviceToHost);
	}
	else
	{
		memcpy(dst->data, src->data, size);
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


MatrixCudaType Matrix::selectUseCuda(Matrix* A1 /*= nullptr*/, Matrix* A2 /*= nullptr*/, Matrix* A3 /*= nullptr*/, Matrix* A4 /*= nullptr*/)
{
	Matrix* m[4] = { A1,A2,A3,A4 };
	for (int i = 0; i < 4; i++)
	{
		if (m[i] && m[i]->UseCuda == mc_NoCuda)
			return mc_NoCuda;
	}
	return globalUseCuda;
}

//矩阵乘，R = aAB+cR
void Matrix::product(Matrix* A, Matrix* B, Matrix* R,
	real a /*= 1*/, real c /*= 0*/, MatrixTransType ta /*= NoTrans*/, MatrixTransType tb /*= NoTrans*/)
{
	int m = R->row;
	int n = R->col;
	int lda = A->row;
	int k = A->col;
	int ldb = B->row;
	if (ta == mt_Trans) { k = A->row; }
	if (R->UseCuda == mc_UseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		auto tb1 = get_cublas_trans(tb);
		CUBLAS_FUNC(gemm)(cublasHandle, ta1, tb1, m, n, k, &a, A->data, lda, B->data, ldb, &c, R->data, m);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		auto tb1 = get_cblas_trans(tb);
		CBLAS_FUNC(gemm)(CblasColMajor, ta1, tb1, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
	}
}

//矩阵乘以向量，R = aAB+cR
void Matrix::productVector(Matrix* A, Matrix* B, Matrix* R, real a /*= 1*/, real c /*= 0*/, MatrixTransType ta /*= NoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == mt_Trans) { std::swap(m, n); };

	if (R->UseCuda == mc_UseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		CUBLAS_FUNC(gemv)(cublasHandle, ta1, m, n, &a, A->data, A->row, B->data, 1, &c, R->data, 1);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		CBLAS_FUNC(gemv)(CblasColMajor, ta1, m, n, a, A->data, A->row, B->data, 1, c, R->data, 1);
	}
}

//没什么用，废弃
void Matrix::productVector2(Matrix* A, Matrix* B, Matrix* R, real a /*= 1*/, real c /*= 0*/, MatrixTransType ta /*= NoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == mt_Trans) { std::swap(m, n); };

	if (R->UseCuda == mc_UseCuda)
	{
		auto ta1 = get_cublas_trans(ta);
		for (int i = 0; i <= R->col; i++)
			CUBLAS_FUNC(gemv)(cublasHandle, ta1, m, n, &a, A->data, A->row, B->data, 1, &c, R->getDataPointer(0, i), 1);
	}
	else
	{
		auto ta1 = get_cblas_trans(ta);
		for (int i = 0; i <= R->col; i++)
			CBLAS_FUNC(gemv)(CblasColMajor, ta1, m, n, a, A->data, A->row, B->data, 1, c, R->getDataPointer(0, i), 1);
	}
}

//矩阵元素乘
void Matrix::hadamardProduct(Matrix* A, Matrix* B, Matrix* R)
{
	if (R->UseCuda == mc_UseCuda)
	{
		real a1 = 1, a2 = 1, b = 0;
		cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_real, CUDNN_NOT_PROPAGATE_NAN);
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
	if (R->UseCuda == mc_UseCuda)
	{
		real a = -1;
		CUBLAS_FUNC(copy)(cublasHandle, R->max_script, A->data, 1, R->data, 1);
		CUBLAS_FUNC(axpy)(cublasHandle, R->max_script, &a, B->data, 1, R->data, 1);

		//real a1 = 1, a2 = -1, b = 0;
		//setTensor(td, 1, 1, R->col, R->row);
		//cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN);
		//cudnnOpTensor(cudnnHandle, od, &a1, td, A->data, &a2, td, B->data, &b, td, R->data);
	}
	else
	{
		CBLAS_FUNC(copy)(R->max_script, A->data, 1, R->data, 1);
		CBLAS_FUNC(axpy)(R->max_script, -1, B->data, 1, R->data, 1);

		// #pragma loop(hint_parallel(8))
		// 	for (int i = 0; i < R->max_script; i++)
		// 	{
		// 		R->data[i] = A->data[i] - B->data[i];
		// 	}
	}
}


real* Matrix::mallocData(int size)
{
	if (UseCuda == mc_UseCuda)
	{
		real* d = nullptr;
		if (cudaMalloc((void **)&d, size * sizeof(real)) == cudaSuccess)
		{
			//dataIsWhere = DataInDevice;
		}
		return d;
	}
	else
	{
		return new real[size];
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

real* Matrix::malloc_getDataFromDevice()
{
	if (UseCuda == mc_UseCuda)
	{
		auto temp = new real[max_script];
		cudaMemcpy(temp, data, sizeof(real)*max_script, cudaMemcpyDeviceToHost);
		return temp;
	}
	else
	{
		return data;
	}
}

void Matrix::freeDataForDevice(real* temp)
{
	if (UseCuda == mc_UseCuda)
	{
		delete temp;
	}
}

real* Matrix::mallocDataForDevice()
{
	if (UseCuda == mc_UseCuda)
	{
		return new real[max_script];
	}
	else
	{
		return data;
	}
}

void Matrix::set_freeDataToDevice(real* temp)
{
	if (UseCuda == mc_UseCuda)
	{
		cudaMemcpy(data, temp, sizeof(real)*max_script, cudaMemcpyHostToDevice);
		delete temp;
	}
}

void Matrix::setTensorDes(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w)
{
	if (tensor && globalUseCuda == mc_UseCuda)
	{
		cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_real, n, c, h, w);
	}
}

//池化，注意利用一个record记录下了对应位置
//gpu部分，平均模式下对padding的支持目前还有问题
void Matrix::poolingForward(ResampleType re, Matrix* X, Matrix* Y,
	int window_w, int window_h, int stride_w, int stride_h, int* recordPos /*= nullptr*/)
{
	if (Y->UseCuda == mc_UseCuda)
	{
		real a = 1, b = 0;
		cudnnSetPooling2dDescriptor(pd, cudnnPoolingMode_t(re), CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
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
					real v = 0;
					//if (re == re_Average)v = 0;
					if (re == re_Max) v = -REAL_MAX;
					int n = 0;
					for (int i_X = i_Y*stride_w; i_X < std::min(X->W, i_Y*stride_w + window_w); i_X++)
					{
						for (int j_X = j_Y*stride_h; j_X < std::min(X->H, j_Y*stride_h + window_h); j_X++)
						{
							if (re == re_Average_Padding || re == re_Average_NoPadding)
							{
								v += X->getData(i_X, j_X, p);
								if(recordPos) recordPos[i_X + j_X*X->W + p*X->H*X->W] = i_Y + j_Y*Y->W + p*Y->H*Y->W;
								n++;
							}
							else if (re == re_Max)
							{
								auto x = X->getData(i_X, j_X, p);
								if (x > v)
								{
									v = x;
									if (recordPos) recordPos[i_Y + j_Y*Y->W + p*Y->H*Y->W] = i_X + j_X*X->W + p*X->H*X->W;
								}
							}
						}
					}
					if (re == re_Average_Padding)
					{
						v /= window_w*window_h;
					}
					else if (re == re_Average_NoPadding)
					{
						v /= n;
					}
					Y->getData(i_Y, j_Y, p) = v;
				}
			}
		}
	}
}

//使用cpu时利用了record
void Matrix::poolingBackward(ResampleType re, Matrix* Y, Matrix* dY, Matrix* X, Matrix* dX,
	int window_w, int window_h, int stride_w, int stride_h, int* recordPos /*= nullptr*/)
{
	if (dX->UseCuda == mc_UseCuda)
	{
		//这个怎么看都快不了
		real a = 1, b = 0;
		cudnnSetPooling2dDescriptor(pd, cudnnPoolingMode_t(re), CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
		cudnnPoolingBackward(cudnnHandle, pd, &a, Y->tensorDes, Y->data, dY->tensorDes, dY->data, X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
	}
	else
	{
		if (re == re_Max && recordPos)
		{
			//cpu计算时必须传入一个记录数组，保存最大值的位置，这样速度会快一点
			dX->initData(0);
			for (int i = 0; i < dY->getDataCount(); i++)
			{
				dX->getData(recordPos[i]) = dY->getData(i);
			}
		}
		else if (re == re_Average_Padding && recordPos)
		{
			for (int i = 0; i < dX->getDataCount(); i++)
			{
				dX->getData(i) = dY->getData(recordPos[i]) / window_w / window_h;
			}
		}
		else if ((re == re_Average_Padding && recordPos == nullptr) || re == re_Average_NoPadding)
		{
			//以下两种算法实际上遍历元素的数目是相同的
			for (int p = 0; p < dY->N*dY->C; p++)
			{
				for (int i_DY = 0; i_DY < dY->W; i_DY++)
				{
					for (int j_DY = 0; j_DY < dY->H; j_DY++)
					{
						int n;						
						if (re == re_Average_NoPadding)
						{
							n = std::min(window_w, dX->W - i_DY*stride_w) * std::min(window_h,dX->H - j_DY*stride_h);
						}
						else
						{
							n = window_w * window_h;
						}
						real v = dY->getData(i_DY, j_DY, p) / n;
						for (int i_DX = i_DY*stride_w; i_DX < std::min(dX->W, i_DY*stride_w + window_w); i_DX++)
						{
							for (int j_DX = j_DY*stride_h; j_DX < std::min(dX->H, j_DY*stride_h + window_h); j_DX++)
							{
								dX->getData(i_DX, j_DX, p) = v;
							}
						}
					}
				}
			}
		}
	}
}

void Matrix::convolutionForward(Matrix* X, Matrix* conv_kernel, Matrix* Y, int m_subA, int n_subA, int m_subR, int n_subR, int countPerGroup)
{
	if (Y->UseCuda == mc_UseCuda)
	{
	}
	else
	{
		for (int p = 0; p < Y->N*Y->C; p++)
		{
			for (int i_R = 0; i_R < Y->W; i_R++)
			{
				for (int j_R = 0; j_R < Y->H; j_R++)
				{
					real v = 0;
					for (int i_A = i_R; i_A < std::min(X->W, i_R + conv_kernel->row); i_A++)
					{
						for (int j_A = j_R; j_A < std::min(X->H, j_R + conv_kernel->col); j_A++)
						{
							v += X->getData(i_A, j_A, p) * conv_kernel->getData(i_A - i_R, j_A - j_R);
						}
					}
					Y->getData(i_R, j_R, p) = v;
				}
			}
		}
	}
}

//这里应该有优化的办法，再说
void Matrix::selectFunction(MatrixCudaType useCuda, real* x, real* y, int size,
	std::function<int(real*, real*, int)> f1, std::function<int(real*, real*, int)> f2)
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



void Matrix::setActive(cudnnActivationMode_t am)
{
	cudnnSetActivationDescriptor(ad, am, CUDNN_NOT_PROPAGATE_NAN, 1);
}

void Matrix::activeForward(ActiveFunctionType af, Matrix* X, Matrix* Y)
{
	real a = 1, b = 0;
	MatrixCudaType useCuda = Y->UseCuda;
	// 	if (X->UseCuda != mc_UseCuda || Y->UseCuda != mc_UseCuda)
	// 	{
	// 		useCuda = mc_NoCuda;
	// 	}
	switch (af)
	{
	case af_Sigmoid:
		if (useCuda == mc_UseCuda)
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
		if (useCuda == mc_UseCuda)
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
				real sum = Y->sumColAbs(i);
				if (sum == 0) continue;
				Y->colMultiply(1 / sum, i);
			}
		}
		break;
	case af_Tanh:
		if (useCuda == mc_UseCuda)
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
		if (Y->max_script <= 0) return;
		Y->initData(0);
		if (useCuda == mc_UseCuda)
		{
			auto T = new Matrix(Y->row, Y->col, md_Inside, mc_NoCuda);
			for (int i_group = 0; i_group < Y->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				T->getData(index, i_group) = 1;
			}
			cpyData(Y, T);
			delete T;
		}
		else
		{
			for (int i_group = 0; i_group < Y->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				Y->getData(index, i_group) = 1;
			}
		}
		break;
	case af_Softplus:
		if (useCuda == mc_UseCuda)
		{

		}
		else
		{
			MyMath::softplus_v(X->data, Y->data, Y->max_script);
		}
		break;
	case af_ReLU:
		if (useCuda == mc_UseCuda)
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

void Matrix::activeBackward(ActiveFunctionType af, Matrix* Y, Matrix* X, Matrix* dX)
{
	real a = 1, b = 0;
	MatrixCudaType useCuda = dX->UseCuda;
	// 	if (X->UseCuda != mc_UseCuda || Y->UseCuda != mc_UseCuda || dX->UseCuda != mc_UseCuda)
	// 	{
	// 		useCuda = mc_NoCuda;
	// 	}
	switch (af)
	{
	case af_Sigmoid:
		if (useCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			cudnnActivationBackward(cudnnHandle, ad, &a, Y->tensorDes, Y->data, dX->tensorDes, dX->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			//MyMath::sigmoid_vb(X->data, dX->data, dX->max_script);
			MyMath::sigmoid_vb2(Y->data, dX->data, dX->max_script);
		}
		break;
	case af_Linear:
		dX->initData(1);
		break;
	case af_Softmax:
		//softmax一般是最后一层，可能无用
		if (useCuda == mc_UseCuda)
		{
			setTensorDes(td, X->col, 1, 1, X->row);
			//TODO: wei wan cheng
			//cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
			//	&alpha, td, A->data, &beta, td, R->data);
		}
		else
		{
			MyMath::exp_v(X->data, dX->data, dX->max_script);
		}
		break;
	case af_Tanh:
		if (useCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_TANH);
			cudnnActivationBackward(cudnnHandle, ad, &a, Y->tensorDes, Y->data, dX->tensorDes, dX->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			MyMath::tanh_vb(X->data, dX->data, dX->max_script);
		}
		break;
	case af_Findmax:
		if (useCuda == mc_UseCuda)
		{

		}
		else
		{

		}
		break;
	case af_Softplus:
		if (useCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_SIGMOID);
			cudnnActivationForward(cudnnHandle, ad, &a, X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			MyMath::softplus_vb(X->data, dX->data, dX->max_script);
		}
		break;
	case af_ReLU:
		if (useCuda == mc_UseCuda)
		{
			setActive(CUDNN_ACTIVATION_RELU);
			cudnnActivationBackward(cudnnHandle, ad, &a, Y->tensorDes, Y->data, dX->tensorDes, dX->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			MyMath::relu_vb(X->data, dX->data, dX->max_script);
		}
		break;
	}
}

