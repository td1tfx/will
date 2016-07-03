#pragma once
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include "lib/cblas.h"
#include "types.h"
#include "MyMath.h"
#include "MyCudaMath.h"

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

struct Matrix
{
private:
	static bool inited;

	MatrixCudaType UseCuda = mc_NoCuda;
	static MatrixCudaType globalUseCuda;

	double* data = nullptr;
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
	double& getData(int w, int h, int p) { return data[w+h*W+p*W*H]; }
	double& getData(int w, int h, int c, int n) { return getData(w,h,0); }

	//这两个不推荐使用，比较乱
	double& getImageData(int m, int n) { return getData(n, m); }
	double* getImageDataPointer(int m, int n) { return &getData(n, m); }

	//这个函数可能不安全，慎用！！
	void resetDataPointer(double* d, int d_in_cuda = 0);
	//使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
	void setInsideData(MatrixDataType id) { insideData = id; }

	double& operator [] (int i) { return data[i]; }

	static void initCuda();
	static void destroyCuda();

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

	static void cpyData(Matrix* dst, Matrix* src);
	void tryUploadToCuda();
	void tryDownloadFromCuda();
	void shareData(Matrix* A, int m, int n);

	static void product(Matrix* A, Matrix* B, Matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans, MatrixTransType tb = mt_NoTrans);
	static void productVector(Matrix* A, Matrix* B, Matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans);
	static void productVector2(Matrix* A, Matrix* B, Matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans);
	static void hadamardProduct(Matrix* A, Matrix* B, Matrix* R);
	static void minus(Matrix* A, Matrix* B, Matrix* R);

	static void pooling(Matrix* A, Matrix* R, int m_subA, int n_subA, int m_subR, int n_subR,
		int countPerGroup, ResampleType re, int** maxPos = nullptr);
	static void convolution(Matrix* A, Matrix* conv_kernel, Matrix* R);
	static void convolution_colasImage(Matrix* A, Matrix* conv_kernel, Matrix* R,
		int m_subA, int n_subA, int m_subR, int n_subR, int countPerGroup);

private:
	static cublasHandle_t cublasHandle;
	static cudnnHandle_t cudnnHandle;
	static cublasOperation_t get_cublas_trans(MatrixTransType t) { return t == mt_NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; }
	static CBLAS_TRANSPOSE get_cblas_trans(MatrixTransType t) { return t == mt_NoTrans ? CblasNoTrans : CblasTrans; }

	cudnnTensorDescriptor_t tensorDes;

	static cudnnTensorDescriptor_t td;
	static cudnnActivationDescriptor_t ad;
	static cudnnOpTensorDescriptor_t od;
	static cudnnPoolingDescriptor_t pd;
	static cudnnConvolutionDescriptor_t cd;

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
	static void selectFunction(MatrixCudaType useCuda, double* x, double* y, int size,
		std::function<int(double*, double*, int)> f1, std::function<int(double*, double*, int)> f2);

	static void setTensor(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w);
	static void setActive(cudnnActivationMode_t am);
	static void setActiveParameter(cudnnActivationMode_t am, int n, int c, int h, int w);
	static void activeForward(ActiveFunctionType af, Matrix* A, Matrix* R);
	static void activeBackward(ActiveFunctionType af, Matrix* A, Matrix* B, Matrix* R);

};

