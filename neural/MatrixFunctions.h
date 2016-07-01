#pragma once
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include "lib/cblas.h"
#include "MyMath.h"

#ifdef _MSC_VER
#define _USE_CUDA
#endif

#ifdef _USE_CUDA
#include "lib/cublas/cuda_runtime.h"
#include "lib/cublas/cublas_v2.h"
#include "lib/cublas/helper_cuda.h"
#include "myth_cuda.h"

#pragma comment (lib, "cublas.lib")
#pragma comment (lib, "cudart_static.lib")
#ifdef _DEBUG
#pragma comment (lib, "neural-cuda.lib")
#else
#pragma comment (lib, "neural-cuda.lib")
#endif

#else

//屏蔽所有cuda函数

#define cublasHandle_t int

#define cublasOperation_t CBLAS_TRANSPOSE
#define CUBLAS_OP_N CblasNoTrans
#define CUBLAS_OP_T CblasTrans
#define cudaMemcpyDeviceToDevice 0
#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyHostToDevice 0
#define cudaSuccess 0

#define cudaMalloc
#define cudaFree
#define cudaMemcpy

#define cublasIdamax
#define cublasDasum
#define cublasDdot
#define cublasDscal
#define cublasDgemm
#define cublasDgemv
#define cublasDcopy
#define cublasDaxpy

#define cuda_exp
#define cuda_sigmoid
#define cuda_dsigmoid
#define cuda_hadamardProduct

#endif

typedef enum
{
	ms_ColMajor,
	ms_RowMajor,
} MatrixStoreType;

typedef enum
{
	mt_NoTrans,
	mt_Trans,
} MatrixTransType;

struct Position
{
	int m, n;
};

struct d_matrix
{
private:
	static bool inited;

	int UseCuda = false;
	static int globalUseCuda;

	double* data = nullptr;
	int row = 0;
	int col = 0;
	int max_script;
	int insideData = 1;
	int data_size = -1;

	//一列的数据作为一个或一组图像，矩阵本身是列优先
	//但是在图片处理，包含卷积核默认是行优先，也就是说图片和卷积核可以认为是转置保存的！！

public:
	d_matrix(int m, int n, int tryInsideData = 1, int tryUseCuda = 1);
	~d_matrix();
	int getRow() { return row; }
	int getCol() { return col; }
	int getDataCount() { return max_script; }
	int xy2i(int m, int n) { return m + n*row; }
	double& getData(int m, int n) { return data[std::min(xy2i(m, n), max_script - 1)]; }
	double& getData(int i) { return data[std::min(i, max_script - 1)]; }
	double* getDataPointer(int m, int n) { return &getData(m, n); }
	double* getDataPointer(int i) { return &getData(i); }
	double* getDataPointer() { return data; }
	int resize(int m, int n, int force = 0);

	//这两个不推荐使用，比较乱
	double& getImageData(int m, int n) { return getData(n, m); }
	double* getImageDataPointer(int m, int n) { return &getData(n, m); }

	//这个函数可能不安全，慎用！！
	void resetDataPointer(double* d, int d_in_cuda = 0);
	//使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
	void setInsideData(int id) { insideData = id; }

	double& operator [] (int i)
	{
		return data[i];
	}
	static CBLAS_TRANSPOSE get_cblas_trans(MatrixTransType t)
	{
		return t == mt_NoTrans ? CblasNoTrans : CblasTrans;
	}

	static void initCublas();

	void print(FILE* fout = stdout);
	int load(double* v, int n);
	void printAsVector(FILE* fout = stdout);
	int loadAsVector(double* v, int n);

	void memcpyDataIn(double* src, int size);
	void memcpyDataOut(double* dst, int size);
	void expand();
	int indexColMaxAbs(int c);
	double sumColAbs(int c);
	double ddot();

	void initData(double v);
	void initRandom();
	void initInt();
	void multiply(double v);
	void colMultiply(double v, int c);

	static void cpyData(d_matrix* dst, d_matrix* src);
	void tryUploadToCuda();
	void tryDownloadFromCuda();
	void shareData(d_matrix* A, int m, int n);

	static void product(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans, MatrixTransType tb = mt_NoTrans);
	static void productVector(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans);
	static void productVector2(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans);
	static void hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R);
	static void minus(d_matrix* A, d_matrix* B, d_matrix* R);

	static void resample(d_matrix* A, d_matrix* R, ResampleType re, int** maxPos, int basePos);
	static void resample_colasImage(d_matrix* A, d_matrix* R, int m_subA, int n_subA, int m_subR, int n_subR,
		int countPerGroup, ResampleType re, int** maxPos = nullptr);
	static void convolution(d_matrix* A, d_matrix* conv_kernel, d_matrix* R);
	static void convolution_colasImage(d_matrix* A, d_matrix* conv_kernel, d_matrix* R, int m_subA, int n_subA, int m_subR, int n_subR, int countPerGroup);

private:
	static cublasHandle_t handle;
	static cublasOperation_t get_cublas_trans(MatrixTransType t)
	{
		return t == mt_NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
	}

	//必须配对！
	double* mallocData(int size);
	void freeData();

	//必须配对！
	double* malloc_getDataFromDevice();
	void freeDataForDevice(double* temp);
	//必须配对！
	double* mallocDataForDevice();
	void set_freeDataToDevice(double* temp);

public:
	void activeFunction(ActiveFunctionType af) { activeFunction(this, this, af); }
	void dactiveFunction(ActiveFunctionType af) { dactiveFunction(this, this, af); }
	static void activeFunction(d_matrix* A, d_matrix* R, ActiveFunctionType af);
	static void dactiveFunction(d_matrix* A, d_matrix* R, ActiveFunctionType af);

};


