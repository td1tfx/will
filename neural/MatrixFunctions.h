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
	NoTrans,
	Trans,
} d_matrixTrans;

typedef enum
{
	DataInHost,
	DataInDevice,
	DataInBoth,
	DataInNowhere,
} DataPosition;

typedef enum
{
	Sigmoid = 0,
	Softmax,
	Tanh,
	Findmax,
} ActiveFunctionMode;

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
	DataPosition dataIsWhere = DataInHost;
	int data_size = -1;

	//一列的数据作为一个或一组图像，矩阵本身是列优先
	//但是在图片处理，包含卷积核默认是行优先，也就是说图片和卷积核可以认为是转置保存的！！
	int col_image_count;
	d_matrix* image = nullptr;

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

	void initColImage(int width, int height, int count);
	void setColImage(int i, int col);
	d_matrix* getColImage();
	
	//这两个不推荐使用，比较乱
	double& getImageData(int m, int n) { return getData(n, m); }
	double* getImageDataPointer(int m, int n) { return &getData(n, m); }

	//这个函数可能不安全，慎用！！
	void resetDataPointer(double* d, int d_in_cuda=0);
	//使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
	void setInsideData(int id) { insideData = id; }

	double& operator [] (int i)
	{
		return data[i];
	}
	static CBLAS_TRANSPOSE get_cblas_trans(d_matrixTrans t)
	{
		return t == NoTrans ? CblasNoTrans : CblasTrans;
	}

	static void initCublas();

	void print(FILE* fout);
	int load(double* v, int n);
	void memcpyDataIn(double* src, int size);
	void memcpyDataOut(double* dst, int size);
	void expand();
	int indexColMaxAbs(int c);
	double sumColAbs(int c);
	double ddot();

	void initData(double v);
	void initRandom();
	void multiply(double v);
	void colMultiply(double v, int c);

	static void cpyData(d_matrix* dst, d_matrix* src);
	void tryUploadToCuda();
	void tryDownloadFromCuda();
	void shareData(d_matrix* A, int m, int n);

	static void product(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, d_matrixTrans ta = NoTrans, d_matrixTrans tb = NoTrans);
	static void productVector(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, d_matrixTrans ta = NoTrans);
	static void productVector2(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, d_matrixTrans ta = NoTrans);
	static void hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R);
	static void minus(d_matrix* A, d_matrix* B, d_matrix* R);

	static void resample(d_matrix* A, d_matrix* R, int mode = 0);
	static void resample2(d_matrix* A, d_matrix* R, int m, int n, int count, int mode = 0);
	static void convolution(d_matrix* A, d_matrix* CORE, d_matrix* R);
	static void convolution2(d_matrix* A, d_matrix* R);

private:
	static cublasHandle_t handle;
	static cublasOperation_t get_cublas_trans(d_matrixTrans t)
	{
		return t == NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
	}
public:

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
	void activeFunction(ActiveFunctionMode afm) { activeFunction(this, this, afm); }
	void dactiveFunction(ActiveFunctionMode afm) { dactiveFunction(this, this, afm); }
	static void activeFunction(d_matrix* A, d_matrix* R, ActiveFunctionMode afm);
	static void dactiveFunction(d_matrix* A, d_matrix* R, ActiveFunctionMode afm);

};


