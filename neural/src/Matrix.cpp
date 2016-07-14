#include "Matrix.h"


MatrixCudaType Matrix::globalUseCuda = mc_NoCuda;
bool Matrix::inited = false;

cublasHandle_t Matrix::cublasHandle;
cudnnHandle_t Matrix::cudnnHandle;
cudnnTensorDescriptor_t Matrix::TensorDes;
cudnnActivationDescriptor_t Matrix::ActivationDes;
cudnnOpTensorDescriptor_t Matrix::OpTensorDes;
cudnnPoolingDescriptor_t Matrix::PoolingDes;
cudnnConvolutionDescriptor_t Matrix::ConvolutionDes;
cudnnFilterDescriptor_t Matrix::FilterDes;
cudnnRNNDescriptor_t Matrix::RNNDes;
cudnnDropoutDescriptor_t Matrix::DropoutDes;
cudnnSpatialTransformerDescriptor_t Matrix::SpatialTransformerDes;

void* Matrix::workspace;

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
//当矩阵是张量时，实际上原本的列数毫无意义，这样写是为了n和c都是1的情况下与原矩阵等价
Matrix::Matrix(int w, int h, int c, int n, MatrixDataType tryInside /*= md_Inside*/, MatrixCudaType tryCuda /*= mc_UseCuda*/)
	:Matrix(w, h*n*c, tryInside, tryCuda)
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
	//fprintf(stderr, "~Matrix\n");
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

	cudnnCreateTensorDescriptor(&TensorDes);
	cudnnCreateActivationDescriptor(&ActivationDes);
	cudnnCreateOpTensorDescriptor(&OpTensorDes);
	cudnnCreatePoolingDescriptor(&PoolingDes);
	cudnnCreateConvolutionDescriptor(&ConvolutionDes);
	cudnnCreateFilterDescriptor(&FilterDes);
	cudnnCreateRNNDescriptor(&RNNDes);
	cudnnCreateDropoutDescriptor(&DropoutDes);
	cudnnCreateSpatialTransformerDescriptor(&SpatialTransformerDes);
	cudaMalloc(&workspace, workspace_size);
#endif	
}

void Matrix::destroyCuda()
{
	if (!inited) return;
	inited = false;
	globalUseCuda = mc_NoCuda;
#ifdef _USE_CUDA
	cudnnDestroyTensorDescriptor(TensorDes);
	cudnnDestroyActivationDescriptor(ActivationDes);
	cudnnDestroyOpTensorDescriptor(OpTensorDes);
	cudnnDestroyPoolingDescriptor(PoolingDes);
	cudnnDestroyConvolutionDescriptor(ConvolutionDes);
	cudnnDestroyFilterDescriptor(FilterDes);
	cudnnDestroyRNNDescriptor(RNNDes);
	cudnnDestroyDropoutDescriptor(DropoutDes);
	cudnnDestroySpatialTransformerDescriptor(SpatialTransformerDes);
	cudaFree(workspace);
	cublasDestroy(cublasHandle);
	cudnnDestroy(cudnnHandle);
#endif
}

void Matrix::print(FILE* fout)
{
	auto temp = malloc_getDataFromDevice();
	// 	for (int p = 0; p < C*N*W*H; p++)
	// 	{
	// 		fprintf(fout, "%14.11f ", temp[p]);
	// 	}
	// 	fprintf(fout, "\n");
	for (int p = 0; p < C*N; p++)
	{
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				auto v = temp[whp2i(w, h, p)];
				if (std::abs(v) > 1e10)
					fprintf(fout, "%14.11e ", v);
				else
					fprintf(fout, "%14.11f ", v);
			}
			fprintf(fout, "\n");
		}
		fprintf(fout, "\n");
	}
	freeDataForDevice(temp);
}

int Matrix::load(real* v, int n)
{
	auto temp = mallocDataForDevice();
	int k = 0;
	for (int p = 0; p < C*N; p++)
	{
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				temp[whp2i(w, h, p)] = v[k++];
				if (k >= n) break;
			}
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
void Matrix::memcpyDataInFromHost(real* src, int size)
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
void Matrix::memcpyDataOutToHost(real* dst, int size)
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

//绝对值求和（直接调用的blas，注意这里实际上需要的功能只是求和）
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
//inc不为零时仅用于测试，不要用于实际计算！
void Matrix::initData(real v, int inc/*=0*/)
{
	if (UseCuda == mc_UseCuda && inc == 0)
	{
		if (UseCuda == mc_UseCuda)
		{
			setTensorDes(TensorDes, 1, 1, col, row);
			cudnnSetTensor(cudnnHandle, TensorDes, data, &v);
		}
	}
	else
	{
		auto temp = mallocDataForDevice();
		//#pragma loop(hint_parallel(8))
		for (int i = 0; i < max_script; i++)
		{
			temp[i] = i*inc + v;
		}
		set_freeDataToDevice(temp);
	}
}


//随机数初始化矩阵，注意这个函数调用次数很少
void Matrix::initRandom()
{
	Random<real> r;
	//r.set_uniform(0, 1);
	auto temp = mallocDataForDevice();
	//#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		//temp[i] = 2.0 * r.rand_uniform() - 1;
		temp[i] = r.rand_normal();
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
		cudnnSetOpTensorDescriptor(OpTensorDes, CUDNN_OP_TENSOR_MUL, MYCUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN);
		cudnnOpTensor(cudnnHandle, OpTensorDes, &a1, A->tensorDes, A->data, &a2, B->tensorDes, B->data, &b, R->tensorDes, R->data);
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
void Matrix::add(Matrix* A, real b, Matrix* B, Matrix* R)
{
	if (R->UseCuda == mc_UseCuda)
	{
		CUBLAS_FUNC(copy)(cublasHandle, R->max_script, A->data, 1, R->data, 1);
		CUBLAS_FUNC(axpy)(cublasHandle, R->max_script, &b, B->data, 1, R->data, 1);

		//real a1 = 1, a2 = -1, b = 0;
		//setTensor(td, 1, 1, R->col, R->row);
		//cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN);
		//cudnnOpTensor(cudnnHandle, od, &a1, td, A->data, &a2, td, B->data, &b, td, R->data);
	}
	else
	{
		CBLAS_FUNC(copy)(R->max_script, A->data, 1, R->data, 1);
		CBLAS_FUNC(axpy)(R->max_script, b, B->data, 1, R->data, 1);

		// #pragma loop(hint_parallel(8))
		// 	for (int i = 0; i < R->max_script; i++)
		// 	{
		// 		R->data[i] = A->data[i] - B->data[i];
		// 	}
	}
}


real Matrix::dot(Matrix* A, int cA, Matrix* B, int cB)
{
	if (A->UseCuda == mc_UseCuda)
	{
		real r;
		CUBLAS_FUNC(dot)(cublasHandle, A->row, A->getDataPointer(0, cA), 1, B->getDataPointer(0, cA), 1, &r);
		return r;

	}
	else
	{
		return CBLAS_FUNC(dot)(A->row, A->getDataPointer(0, cA), 1, B->getDataPointer(0, cA), 1);
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
		cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, n, c, h, w);
	}
}

//池化，注意利用一个record记录下了对应位置
//gpu部分，平均模式下对padding的支持目前还有问题
void Matrix::poolingForward(ResampleType re, Matrix* X, Matrix* A,
	int window_w, int window_h, int stride_w, int stride_h, int* recordPos /*= nullptr*/)
{
	if (A->UseCuda == mc_UseCuda)
	{
		real a = 1, b = 0;
		cudnnSetPooling2dDescriptor(PoolingDes, cudnnPoolingMode_t(re), CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
		cudnnPoolingForward(cudnnHandle, PoolingDes, &a, X->tensorDes, X->data, &b, A->tensorDes, A->data);
	}
	else
	{
		for (int p = 0; p < A->N*A->C; p++)
		{
			for (int wA = 0; wA < A->W; wA++)
			{
				for (int hA = 0; hA < A->H; hA++)
				{
					real v = 0;
					//if (re == re_Average)v = 0;
					if (re == re_Max) v = -REAL_MAX;
					int n = 0;
					for (int wX = wA*stride_w; wX < std::min(X->W, wA*stride_w + window_w); wX++)
					{
						for (int hX = hA*stride_h; hX < std::min(X->H, hA*stride_h + window_h); hX++)
						{
							if (re == re_Average_Padding || re == re_Average_NoPadding)
							{
								v += X->getData(wX, hX, p);
								if (recordPos) recordPos[wX + hX*X->W + p*X->H*X->W] = wA + hA*A->W + p*A->H*A->W;
								n++;
							}
							else if (re == re_Max)
							{
								auto x = X->getData(wX, hX, p);
								if (x > v)
								{
									v = x;
									if (recordPos) recordPos[wA + hA*A->W + p*A->H*A->W] = wX + hX*X->W + p*X->H*X->W;
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
					A->getData(wA, hA, p) = v;
				}
			}
		}
	}
}

//使用cpu时利用了record
void Matrix::poolingBackward(ResampleType re, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
	int window_w, int window_h, int stride_w, int stride_h, int* recordPos /*= nullptr*/)
{
	if (dX->UseCuda == mc_UseCuda)
	{
		//这个怎么看都快不了
		real a = 1, b = 0;
		cudnnSetPooling2dDescriptor(PoolingDes, cudnnPoolingMode_t(re), CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
		cudnnPoolingBackward(cudnnHandle, PoolingDes, &a, A->tensorDes, A->data, dA->tensorDes, dA->data, X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
	}
	else
	{
		if (re == re_Max && recordPos)
		{
			//cpu计算时必须传入一个记录数组，保存最大值的位置，这样速度会快一点
			dX->initData(0);
			for (int i = 0; i < dA->getDataCount(); i++)
			{
				dX->getData(recordPos[i]) = dA->getData(i);
			}
		}
		//对于平均值池化，两种算法实际上遍历元素的数目是相同的
		else if (re == re_Average_Padding && recordPos)
		{
			for (int i = 0; i < dX->getDataCount(); i++)
			{
				dX->getData(i) = dA->getData(recordPos[i]) / window_w / window_h;
			}
		}
		else if ((re == re_Average_Padding && recordPos == nullptr) || re == re_Average_NoPadding)
		{
			for (int p = 0; p < dA->N*dA->C; p++)
			{
				for (int wdA = 0; wdA < dA->W; wdA++)
				{
					for (int hdA = 0; hdA < dA->H; hdA++)
					{
						int n;
						if (re == re_Average_NoPadding)
						{
							n = std::min(window_w, dX->W - wdA*stride_w) * std::min(window_h, dX->H - hdA*stride_h);
						}
						else
						{
							n = window_w * window_h;
						}
						real v = dA->getData(wdA, hdA, p) / n;
						for (int wdX = wdA*stride_w; wdX < std::min(dX->W, wdA*stride_w + window_w); wdX++)
						{
							for (int hdX = hdA*stride_h; hdX < std::min(dX->H, hdA*stride_h + window_h); hdX++)
							{
								dX->getData(wdX, hdX, p) = v;
							}
						}
					}
				}
			}
		}
	}
}

void Matrix::convolutionForward(Matrix* X, Matrix* W, Matrix* A, int* recordX /*= nullptr*/, int* recordW /*= nullptr*/)
{
	if (A->UseCuda == mc_UseCuda)
	{
		cudnnConvolutionFwdAlgoPerf_t cfap[8];
		auto cfa = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
		real a = 1, b = 0;
		int n;
		auto scd = cudnnSetConvolution2dDescriptor(ConvolutionDes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);
		auto sfd = cudnnSetFilter4dDescriptor(FilterDes, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, A->C, X->C, W->H, W->W);
		//cudnnGetConvolutionForwardAlgorithm(cudnnHandle, X->tensorDes, fd, cd, A->tensorDes, 
		//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, workspace_size, &cfa);
		//cudnnFindConvolutionForwardAlgorithm(cudnnHandle, X->tensorDes, fd, cd, A->tensorDes, 8, &n, cfap);
		auto scf = cudnnConvolutionForward(cudnnHandle, &a, X->tensorDes, X->data, FilterDes, W->data, ConvolutionDes,
			cfa, workspace, workspace_size, &b, A->tensorDes, A->data);
		//printf("%d, %d, %d\n", scd, sfd, scf);
	}
	else
	{
		//实际上可以处理为一个大稀疏矩阵乘，太麻烦也不见得会快，不管了
		//除了1CC和CC1，其他的不保证与GPU结果一致
		//if (X->C != 1 && A->C != 1) return;
		A->initData(0);
		convolution_sub(A, W, X, A, W->C, X->N, 1);
	}
}

void Matrix::convolutionBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* W, Matrix* dW, Matrix* dB)
{
	if (dX->UseCuda == mc_UseCuda)
	{
		real a = 1, b = 0;
		int n;
		cudnnSetConvolution2dDescriptor(ConvolutionDes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);
		cudnnStatus_t scbd, scbf, scbb;
		cudnnConvolutionBwdDataAlgoPerf_t cbdap[8];
		cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle, FilterDes, dA->tensorDes, ConvolutionDes, dX->tensorDes, 8, &n, cbdap);
		cudnnConvolutionBwdFilterAlgoPerf_t cbfap[8];
		cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle, X->tensorDes, dA->tensorDes, ConvolutionDes, FilterDes, 8, &n, cbfap);
		if (dX)
		{
			auto cbda = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, FilterDes, dA->tensorDes, ConvolutionDes, dX->tensorDes,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, workspace_size, &cbda);
			scbd = cudnnConvolutionBackwardData(cudnnHandle, &a, FilterDes, W->data, dA->tensorDes, dA->data, ConvolutionDes,
				cbda, workspace, workspace_size, &b, dX->tensorDes, dX->data);
		}
		if (dW)
		{
			auto cbfa = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, X->tensorDes, dA->tensorDes, ConvolutionDes, FilterDes,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, workspace_size, &cbfa);
			scbf = cudnnConvolutionBackwardFilter(cudnnHandle, &a, X->tensorDes, X->data, dA->tensorDes, dA->data, ConvolutionDes,
				cbfa, workspace, workspace_size, &b, FilterDes, dW->data);
		}
		if (dB)
		{
			scbb = cudnnConvolutionBackwardBias(cudnnHandle, &a, dA->tensorDes, dA->data, &b, dB->tensorDes, dB->data);
		}
		//printf("%d, %d, %d\n", scbd, scbf, scbb);
	}
	else
	{
		if (dX)
		{
			dX->initData(0);
			convolution_sub(dA, W, dX, dX, W->C, dX->N, 1);
		}
		if (dW)
		{
			//N不为1情况下不一致
			dW->initData(0);
			convolution_sub(dA, dW, X, dW, dW->C, X->N, 1);
			//dW->multiply(1.0f*dA->N);
		}
		if (dB)
		{
			dB->initData(0);
			//这个好像就是对对应的A求和
			for (int n = 0; n < dA->N; n++)
			{
				for (int c = 0; c < dA->C; c++)
				{
					for (int h = 0; h < dA->H; h++)
					{
						for (int w = 0; w < dA->W; w++)
						{
							dB->getData(0, 0, c, 0) += dA->getData(w, h, c, n);
						}
					}
				}
			}
		}
	}
}

//R必须是XYZ其中之一！XY循环遍历，，坐标运算X在前，Z判断坐标，plus是加减法
//一般来说应选择维度较小的作为X和Y
//只在CPU运算中起作用
void Matrix::convolution_sub(Matrix* X, Matrix* Y, Matrix* Z, Matrix* R, int C, int N, int plus)
{
	if (R->UseCuda == mc_UseCuda) return;

	for (int n = 0; n < N; n++)
	{
		int nX = n % X->N;
		int nY = n % Y->N;
		int nZ = n % Z->N;
		for (int c = 0; c < C; c++)
		{
			int cX = c % X->C;
			int cY = c % Y->C;
			int cZ = c % Z->C;
			for (int wX = 0; wX < X->W; wX++)
			{
				for (int hX = 0; hX < X->H; hX++)
				{
					for (int wY = 0; wY < Y->W; wY++)
					{
						for (int hY = 0; hY < Y->H; hY++)
						{
							int wZ, hZ;
							if (plus == 1)
							{
								wZ = wX + wY;
								hZ = hX + hY;
							}
							else if (plus == -1)
							{
								wZ = wX - wY;
								hZ = hX - hY;
							}
							if (wZ >= 0 && hZ >= 0 && wZ < Z->W && hZ < Z->H)
							{
								if (R == X)
									X->getData(wX, hX, cX, nX) += Y->getData(wY, hY, cY, nY)*Z->getData(wZ, hZ, cZ, nZ);
								else if (R == Y)
									Y->getData(wY, hY, cY, nY) += X->getData(wX, hX, cX, nX)*Z->getData(wZ, hZ, cZ, nZ);
								else if (R == Z)
									Z->getData(wZ, hZ, cZ, nZ) += X->getData(wX, hX, cX, nX)*Y->getData(wY, hY, cY, nY);
							}
						}
					}
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
	cudnnSetActivationDescriptor(ActivationDes, am, CUDNN_NOT_PROPAGATE_NAN, 1);
}

void Matrix::activeForward(ActiveFunctionType af, Matrix* X, Matrix* A, void* data/*= nullptr*/)
{
	real a = 1, b = 0, v = 0;
	auto useCuda = A->UseCuda;
	auto nan = CUDNN_NOT_PROPAGATE_NAN;
	switch (af)
	{
	case af_Sigmoid:
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_SIGMOID, nan, 1);
			cudnnActivationForward(cudnnHandle, ActivationDes, &a, X->tensorDes, X->data, &b, A->tensorDes, A->data);
		}
		else
		{
			VectorMath::sigmoid_v(X->data, A->data, A->max_script);
		}
		break;
	case af_ReLU:
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_RELU, nan, 1);
			cudnnActivationForward(cudnnHandle, ActivationDes, &a, X->tensorDes, X->data, &b, A->tensorDes, A->data);
		}
		else
		{
			VectorMath::relu_v(X->data, A->data, A->max_script);
		}
		break;
	case af_Tanh:
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_TANH, nan, 1);
			cudnnActivationForward(cudnnHandle, ActivationDes, &a, X->tensorDes, X->data, &b, A->tensorDes, A->data);
		}
		else
		{
			VectorMath::tanh_v(X->data, A->data, A->max_script);
		}
		break;
	case af_ClippedReLU:
		if (data) v = *(real*)data;
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_CLIPPED_RELU, nan, v);
			cudnnActivationForward(cudnnHandle, ActivationDes, &a, X->tensorDes, X->data, &b, A->tensorDes, A->data);
		}
		else
		{
			VectorMath::clipped_relu_v(X->data, A->data, A->max_script, v);
		}
		break;
	case af_Softmax:
		if (useCuda == mc_UseCuda)
		{
			setTensorDes(TensorDes, X->col, 1, 1, X->row);
			cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&a, TensorDes, X->data, &b, TensorDes, A->data);
		}
		else
		{
			//因为数值问题，可能需要减去每列最大值
			Matrix::cpyData(A, X);
			for (int i = 0; i < A->col; i++)
			{
				VectorMath::sub_max(A->getDataPointer(0, i), A->row);
			}
			VectorMath::exp_v(A->data, A->data, A->max_script);
			for (int i = 0; i < A->col; i++)
			{
				real sum = A->sumColAbs(i);
				if (sum == 0) continue;
				A->colMultiply(1 / sum, i);
			}
		}
		break;
	case af_SoftmaxLoss:
		if (useCuda == mc_UseCuda)
		{
			setTensorDes(TensorDes, A->col, 1, 1, A->row);
			cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
				&a, TensorDes, X->data, &b, TensorDes, A->data);
		}
		else
		{
			activeForward(af_Softmax, X, A);
			VectorMath::log_v(A->data, A->data, A->max_script);
		}
		break;
	case af_Linear:
		cpyData(A, X);
		break;
	case af_Findmax:
		//计算时尽量不要使用，只用在验证时
		if (A->max_script <= 0) return;
		if (useCuda == mc_UseCuda)
		{
			auto T = new Matrix(A->row, A->col, md_Inside, mc_NoCuda);
			T->initData(0);
			for (int i_group = 0; i_group < A->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				T->getData(index, i_group) = 1;
			}
			cpyData(A, T);
			delete T;
		}
		else
		{
			A->initData(0);
			for (int i_group = 0; i_group < A->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				A->getData(index, i_group) = 1;
			}
		}
		break;
	case af_Softplus:
		//GPU部分不支持
		if (useCuda == mc_UseCuda)
		{

		}
		else
		{
			VectorMath::softplus_v(X->data, A->data, A->max_script);
		}
		break;
	case af_Dropout:
		if (data) v = *(real*)data;
		if (useCuda == mc_UseCuda)
		{
			cudnnSetDropoutDescriptor(DropoutDes, cudnnHandle, v, workspace, workspace_size, 9999);
			cudnnDropoutForward(cudnnHandle, DropoutDes, X->tensorDes, X->data, A->tensorDes, A->data, workspace, workspace_size);
		}
		else
		{
			Random<real> r;
			r.reset();
			for (int i = 0; i < A->max_script; i++)
			{
				if (r.rand_uniform() < v)
				{
					A->data[i] = 0;
				}
				else
				{
					A->data[i] = X->data[i] / (1 - v);
				}
			}
		}
		break;
	}
}

void Matrix::activeBackward(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, void* data /*= nullptr*/)
{
	real a = 1, b = 0, v = 0;
	auto useCuda = dX->UseCuda;
	auto nan = CUDNN_NOT_PROPAGATE_NAN;
	switch (af)
	{
	case af_Sigmoid:
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_SIGMOID, nan, 1);
			cudnnActivationBackward(cudnnHandle, ActivationDes, &a, A->tensorDes, A->data, dA->tensorDes, dA->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			VectorMath::sigmoid_vb(A->data, dA->data, X->data, dX->data, dX->max_script);
		}
		break;
	case af_ReLU:
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_RELU, nan, 1);
			cudnnActivationBackward(cudnnHandle, ActivationDes, &a, A->tensorDes, A->data, dA->tensorDes, dA->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			VectorMath::relu_vb(A->data, dA->data, X->data, dX->data, dX->max_script);
		}
		break;
	case af_Tanh:
		//两者结果在1e-10的精度有区别
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_TANH, nan, 1);
			cudnnActivationBackward(cudnnHandle, ActivationDes, &a, A->tensorDes, A->data, dA->tensorDes, dA->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			VectorMath::tanh_vb(A->data, dA->data, X->data, dX->data, dX->max_script);
		}
		break;
	case af_ClippedReLU:
		if (data) v = *(real*)data;
		if (useCuda == mc_UseCuda)
		{
			cudnnSetActivationDescriptor(ActivationDes, CUDNN_ACTIVATION_CLIPPED_RELU, nan, v);
			cudnnActivationBackward(cudnnHandle, ActivationDes, &a, A->tensorDes, A->data, dA->tensorDes, dA->data,
				X->tensorDes, X->data, &b, dX->tensorDes, dX->data);
		}
		else
		{
			VectorMath::clipped_relu_vb(A->data, dA->data, X->data, dX->data, dX->max_script, v);
		}
		break;
	case af_Softmax:
		if (useCuda == mc_UseCuda)
		{
			setTensorDes(TensorDes, dX->col, 1, 1, dX->row);
			auto s = cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&a, TensorDes, A->data, TensorDes, dA->data, &b, TensorDes, dX->data);
		}
		else
		{
			for (int i = 0; i < dX->col; i++)
			{
				auto v = dot(A, i, dA, i);
				VectorMath::softmax_vb_sub(A->getDataPointer(0, i), dA->getDataPointer(0, i), v, dX->getDataPointer(0, i), dX->row);
			}
		}
		break;
	case af_SoftmaxLoss:
		if (useCuda == mc_UseCuda)
		{
			setTensorDes(TensorDes, X->col, 1, 1, X->row);
			auto s = cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
				&a, TensorDes, A->data, TensorDes, dA->data, &b, TensorDes, dX->data);
		}
		else
		{
			for (int i = 0; i < dX->col; i++)
			{
				real v = 0;
				for (int j = 0; j < dX->row; j++)
				{
					v += dA->getData(i, j);
				}
				VectorMath::softmaxloss_vb_sub(A->getDataPointer(0, i), dA->getDataPointer(0, i), v, dX->getDataPointer(0, i), dX->row);
			}
		}
		break;
	case af_Linear:
		dX->initData(1);
		break;
	case af_Findmax:
		//似乎应该是返回一个常数矩阵，若考虑效率应当留空此处在外部处理
		dX->initData(1);
		break;
	case af_Softplus:
		//该函数导数就是sigmoid
		activeForward(af_Sigmoid, X, dX);
		break;
	case af_Dropout:
		if (data) v = *(real*)data;
		if (useCuda == mc_UseCuda)
		{
			//这个效果与文档中描述不符
			//cudnnSetDropoutDescriptor(DropoutDes, cudnnHandle, v, workspace, workspace_size, 9999);
			cudnnDropoutBackward(cudnnHandle, DropoutDes, dA->tensorDes, dA->data, dX->tensorDes, dX->data, workspace, workspace_size);
		}
		else
		{
			for (int i = 0; i < dX->max_script; i++)
			{
				if (A->data[i] == 0)
				{
					dX->data[i] = 0;
				}
				else
				{
					dX->data[i] = dA->data[i] / (1 - v);
				}
			}
		}
		break;
	}
}

