#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <cfloat>
#include "cblas.h"
#include "types.h"
#include "VectorMath.h"
#include "Random.h"
#include "CudnnTemplate.h"

//列优先或者行优先（未使用）
typedef enum
{
	ms_ColMajor,
	ms_RowMajor,
} MatrixStoreType;

//转置
typedef enum
{
	mt_NoTrans,
	mt_Trans,
} MatrixTransType;

//数据位置（是否需要自己析构数据）
typedef enum
{
	md_Outside = 0,
	md_Inside,
} MatrixDataType;

//数据是否存储于CUDA设备
typedef enum
{
	mc_NoCuda = 0,
	mc_UseCuda,
} MatrixCudaType;

//矩阵和张量
//该类更大的意义是封装cuda运算
struct Matrix
{
private:
	static bool inited;

	MatrixCudaType UseCuda = mc_NoCuda;
	static MatrixCudaType globalUseCuda;

	real* data = nullptr;
	int row = 0;
	int col = 0;
	int max_script;
	MatrixDataType insideData = md_Inside;
	int data_size = -1;

	//一列的数据作为一个或一组图像，矩阵本身是列优先
	//但是在图片处理，包含卷积核默认是行优先，也就是说图片和卷积核可以认为是转置保存的！！
	int W, H, C, N;

public:
	Matrix(int m, int n, MatrixDataType tryInside = md_Inside, MatrixCudaType tryCuda = mc_UseCuda);
	Matrix(int w, int h, int c, int n, MatrixDataType tryInside = md_Inside, MatrixCudaType tryCuda = mc_UseCuda);
	~Matrix();
private:
	int xy2i(int m, int n) { return m + n*row; }
	int whp2i(int w, int h, int p) { return w + h*W + p*W*H; }
public:
	int getRow() { return row; }
	int getCol() { return col; }
	int getDataCount() { return max_script; }
	int getMemerySize() { return max_script * sizeof(real); }
	int whcn2i(int w, int h, int c, int n) { return w + h*W + c*W*H + n*C*W*H; }

	//以下4个函数注意如果数据在显存中，一般来说是无法赋值和输出的
	real& getData(int i) { return data[std::min(i, max_script - 1)]; }
	real& getData(int m, int n) { return data[std::min(xy2i(m, n), max_script - 1)]; }
	real& getData(int w, int h, int p) { return data[whp2i(w, h, p)]; }
	real& getData(int w, int h, int c, int n) { return data[whcn2i(w, h, c, n)]; }

	real* getDataPointer() { return data; }
	real* getDataPointer(int i) { return &getData(i); }
	real* getDataPointer(int m, int n) { return &getData(m, n); }
	real* getDataPointer(int w, int h, int c, int n) { return &getData(w, h, c, n); }
public:
	int resize(int m, int n, int force = 0);

	//重设数据指针，这个函数可能不安全，慎用！！
	void resetDataPointer(real* d) { data = d; }

	//使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
	void setInsideData(MatrixDataType id) { insideData = id; }

	real& operator [] (int i) { return data[i]; }

	static void initCuda();
	static void destroyCuda();

	void print(FILE* fout = stdout);
	int load(real* v, int n);
	void printAsVector(FILE* fout = stdout);
	int loadAsVector(real* v, int n);

	void memcpyDataInFromHost(real* src, int size);
	void memcpyDataOutToHost(real* dst, int size);


	static void cpyData(Matrix* dst, Matrix* src);
	void tryUploadToCuda();
	void tryDownloadFromCuda();
	void shareData(Matrix* A, int m, int n);

	//为静态运算在结果矩阵使用显存时就调用cuda函数计算，但是调用者应保证所有矩阵一致
	//在使用cuda的时候也有可能存在在内存中的矩阵
	
private:
	static const real real_1;
	static const real real_0;

	static cublasHandle_t cublasHandle;
	static cudnnHandle_t cudnnHandle;
	static cublasOperation_t get_cublas_trans(MatrixTransType t) { return t == mt_NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; }
	static CBLAS_TRANSPOSE get_cblas_trans(MatrixTransType t) { return t == mt_NoTrans ? CblasNoTrans : CblasTrans; }

	//为何我感觉这样很恶心
	//一般来说，在XAdXdA中，以X的计算设置为准，反向时一般不再重新设置
	cudnnTensorDescriptor_t TensorDesc = nullptr;
	cudnnTensorDescriptor_t asTensorDesc = nullptr;
	cudnnActivationDescriptor_t ActivationDesc = nullptr;
	cudnnOpTensorDescriptor_t OpTensorDesc = nullptr;
	cudnnPoolingDescriptor_t PoolingDesc = nullptr;
	cudnnConvolutionDescriptor_t ConvolutionDesc = nullptr;
	cudnnFilterDescriptor_t FilterDesc = nullptr;
	cudnnRNNDescriptor_t RNNDesc = nullptr;
	cudnnDropoutDescriptor_t DropoutDesc = nullptr;
	cudnnSpatialTransformerDescriptor_t SpatialTransformerDesc = nullptr;
	cudnnLRNDescriptor_t LRNDesc = nullptr;

	static void* workspace;
	static const int workspace_size = 1024 * 1024 * 128;

	//必须配对！
	real* mallocData(int size);
	void freeData();

	//这两组好像必须交叉配对！
	real* malloc_getDataFromDevice();
	void freeDataForDevice(real* temp);
	real* mallocDataForDevice();
	void set_freeDataToDevice(real* temp);

public:
	//运算函数
	void expand();
	int indexColMaxAbs(int c);
	real sumAbs();
	real sumColAbs(int c);
	real ddot();

	void initData(real v, int inc = 0);
	void initRandom();
	void multiply(real v);
	void colMultiply(real v, int c);

	static void product(Matrix* A, Matrix* B, Matrix* R,
		real a = 1, real c = 0, MatrixTransType ta = mt_NoTrans, MatrixTransType tb = mt_NoTrans);
	static void productVector(Matrix* A, Matrix* B, Matrix* R,
		real a = 1, real c = 0, MatrixTransType ta = mt_NoTrans);
	static void productVector2(Matrix* A, Matrix* B, Matrix* R,
		real a = 1, real c = 0, MatrixTransType ta = mt_NoTrans);
	static void hadamardProduct(Matrix* A, Matrix* B, Matrix* R);
	static void add(Matrix* A, real b, Matrix* B, Matrix* R);
	static real dot(Matrix* A, int cA, Matrix* B, int cB);

	//以下函数不属于矩阵基本运算，因为实在太长，实现分拆到其他文件
	inline static void setTensorDesc(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w);
	inline static void tryInitNullTensorDesc(cudnnTensorDescriptor_t* tensor, int n, int c, int h, int w);

	static void poolingForward(ResampleType re, Matrix* X, Matrix* A,
		int window_w, int window_h, int stride_w, int stride_h, int* recordPos = nullptr);
	static void poolingBackward(ResampleType re, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
		int window_w, int window_h, int stride_w, int stride_h, int* recordPos = nullptr);

	static void convolutionForward(Matrix* X, Matrix* W, Matrix* A, int* recordX = nullptr, int* recordW = nullptr);
	static void convolutionBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* W, Matrix* dW, Matrix* dB);
	static void convolution_sub(Matrix* X, Matrix* Y, Matrix* Z, Matrix* R, int C, int N, int plus);

	static void dropoutForward(Matrix* X, Matrix* A, Matrix* rgStat, Matrix* stat, real v, int seed = 0);
	static void dropoutBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* rgStat, Matrix* stat, real v);

	static void divisiveNormalizationForward(Matrix* X, Matrix* A, 
		Matrix* means, Matrix* temp1, Matrix* temp2, unsigned lrnN, real lrnAlpha, real lrnBeta, real lrnK);
	static void divisiveNormalizationBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, 
		Matrix* means, Matrix* temp1, Matrix* temp2, Matrix* dmeans);

	static void batchNormalizationForward(Matrix* X, Matrix* A, Matrix* rgStat, Matrix* stat);
	static void batchNormalizationBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* rgStat, Matrix* stat, real v);

	static void spatialTfSamplerForward();
	static void spatialTfSamplerBackward();

	//激活和反向激活中，输入和输出矩阵都是同维度
	inline static void setActivationDesc(cudnnActivationDescriptor_t tensor, cudnnActivationMode_t mode, real v);
	inline static void tryInitNullActivationDesc(cudnnActivationDescriptor_t* activation, cudnnActivationMode_t mode, real v);

	static void activeForward(ActiveFunctionType af, Matrix* X, Matrix* A);
	static void activeBackward(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX);

	static void activeForwardEx(ActiveFunctionType af, Matrix* X, Matrix* A,
		std::initializer_list<real> vr_list = { 1 }, std::initializer_list<int> vi_list = { 0 }, std::initializer_list<Matrix*> as_list = {});
	static void activeBackwardEx(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
		std::initializer_list<real> vr_list = { 1 }, std::initializer_list<int> vi_list = { 0 }, std::initializer_list<Matrix*> as_list = {});

};

typedef Matrix Tensor;

