#pragma once
#include "lib/cblas.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>

#ifdef _MSC_VER
#define _USE_CUDA
#endif

#ifdef _USE_CUDA
#include "lib/cublas/cuda_runtime.h"
#include "lib/cublas/cublas_v2.h"
#include "lib/cublas/helper_cuda.h"
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

struct d_matrix
{
private:
	static bool inited;

	int UseCublas = false;
	static int globalUseCublas;

	double* data = nullptr;
	int row = 0;
	int col = 0;
	int max_script;
	int insideData = 1;
	DataPosition dataIsWhere = DataInHost;
	int data_size = -1;
public:
	d_matrix(int x, int y, int tryInsideData = 1, int tryUseCublas = 1);
	~d_matrix() { if (insideData) freeData(); }
	int getRow() { return row; }
	int getCol() { return col; }
	int getDataCount() { return max_script; }
	int xy2i(int x, int y) { return x + y*row; }
	double& getData(int x, int y) { return data[std::min(xy2i(x, y), max_script - 1)]; }
	double& getData(int i) { return data[std::min(i, max_script - 1)]; }
	double* getDataPointer(int x, int y) { return &getData(x, y); }
	double* getDataPointer(int i) { return &getData(i); }
	double* getDataPointer() { return data; }
	void resize(int m, int n, int force = 0);

	//这个函数可能不安全，慎用！！
	void resetDataPointer(double* d);
	//使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
	void setInsideData(bool id) { insideData = id; }

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
	void applyFunction(std::function<double(double)> f);

	static void cpyData(d_matrix* dst, d_matrix* src);

	static void product(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, d_matrixTrans ta = NoTrans, d_matrixTrans tb = NoTrans);
	static void productVector(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, d_matrixTrans ta = NoTrans);
	static void hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R);
	static void minus(d_matrix* A, d_matrix* B, d_matrix* R);
	static void applyFunction(d_matrix* A, d_matrix* R, std::function<double(double)> f);

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

};


