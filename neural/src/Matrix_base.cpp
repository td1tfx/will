#include "Matrix.h"

MatrixCudaType Matrix::globalUseCuda = mc_NoCuda;
bool Matrix::inited = false;

const real Matrix::real_1 = 1;
const real Matrix::real_0 = 0;

cublasHandle_t Matrix::cublasHandle;
cudnnHandle_t Matrix::cudnnHandle;

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
		cudnnCreateTensorDescriptor(&TensorDesc);
	}
	setTensorDesc(TensorDesc, 1, 1, n, m);
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
	setTensorDesc(TensorDesc, n, c, h, w);
}

Matrix::~Matrix()
{
	if (insideData == md_Inside) freeData();

	cudnnDestroyDescriptor(TensorDesc);
	cudnnDestroyDescriptor(asTensorDesc);
	cudnnDestroyDescriptor(ActivationDesc);
	cudnnDestroyDescriptor(OpTensorDesc);
	cudnnDestroyDescriptor(PoolingDesc);
	cudnnDestroyDescriptor(ConvolutionDesc);
	cudnnDestroyDescriptor(FilterDesc);
	cudnnDestroyDescriptor(RNNDesc);
	cudnnDestroyDescriptor(DropoutDesc);
	cudnnDestroyDescriptor(SpatialTransformerDesc);
	cudnnDestroyDescriptor(LRNDesc);
}

//返回值：-1空矩阵，未重新分配内存，1重新分配内存
int Matrix::resize(int m, int n, int force /*= 0*/)
{
	if (!this)
		return -1;
	row = m;
	col = n;
	max_script = m*n;
	setTensorDesc(TensorDesc, 1, 1, n, m);
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
	cudaMalloc(&workspace, workspace_size);
#endif	
}

void Matrix::destroyCuda()
{
	if (!inited) return;
	inited = false;
	globalUseCuda = mc_NoCuda;
#ifdef _USE_CUDA
	cudaFree(&workspace);
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
			cudnnSetTensor(cudnnHandle, TensorDesc, data, &v);
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
		cudnnOpTensorDescriptor_t OpTensorDesc = nullptr;
		CUDNN_CREATE_SET_DESCIPTOR(OpTensorDesc,
			cudnnSetOpTensorDescriptor(OpTensorDesc, CUDNN_OP_TENSOR_MUL, MYCUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN));
		cudnnOpTensor(cudnnHandle, OpTensorDesc, &real_1, A->TensorDesc, A->data, &real_1, B->TensorDesc, B->data, &real_0, R->TensorDesc, R->data);
		cudnnDestroyDescriptor(OpTensorDesc);
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

		//setTensor(td, 1, 1, R->col, R->row);
		//cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN);
		//cudnnOpTensor(cudnnHandle, od, &a1, td, A->data, &a2, td, B->data, &real_0, td, R->data);
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



