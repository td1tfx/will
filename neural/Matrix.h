#pragma once
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include "lib/cblas.h"
#include "types.h"
#include "MyMath.h"
#include "Random.h"

//�����Ȼ��������ȣ�δʹ�ã�
typedef enum
{
	ms_ColMajor,
	ms_RowMajor,
} MatrixStoreType;

//ת��
typedef enum
{
	mt_NoTrans,
	mt_Trans,
} MatrixTransType;

//����λ�ã��Ƿ���Ҫ�Լ��������ݣ�
typedef enum
{
	md_Outside = 0,
	md_Inside,
} MatrixDataType;

//�����Ƿ�洢��CUDA�豸
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

	//һ�е�������Ϊһ����һ��ͼ�񣬾�������������
	//������ͼƬ�������������Ĭ���������ȣ�Ҳ����˵ͼƬ�;���˿�����Ϊ��ת�ñ���ģ���
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
	double& getData(int w, int h, int p) { return data[w + h*W + p*W*H]; }
	double& getData(int w, int h, int c, int n) { return getData(w, h, c + C*n); }

	//���������Ƽ�ʹ�ã��Ƚ���
	double& getImageData(int m, int n) { return getData(n, m); }
	double* getImageDataPointer(int m, int n) { return &getData(n, m); }

	//����������ܲ���ȫ�����ã���
	void resetDataPointer(double* d, int d_in_cuda = 0);
	//ʹ�������������Ҫ��Ϊ������ʱͬʱɾ������ָ�룬�����������ڸ�ɶ��
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
	double sumAbs();
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

	//Ϊ��̬�����ڽ������ʹ���Դ�ʱ�͵���cuda�������㣬���ǵ�����Ӧ��֤���о���һ��
	//��δ����ȫ��UseCuda����Ϊ��ʹ��cuda��ʱ��Ҳ�п��ܴ������ڴ��еľ���
	static MatrixCudaType selectUseCuda(Matrix* A1 = nullptr, Matrix* A2 = nullptr, Matrix* A3 = nullptr, Matrix* A4 = nullptr);

	static void product(Matrix* A, Matrix* B, Matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans, MatrixTransType tb = mt_NoTrans);
	static void productVector(Matrix* A, Matrix* B, Matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans);
	static void productVector2(Matrix* A, Matrix* B, Matrix* R,
		double a = 1, double c = 0, MatrixTransType ta = mt_NoTrans);
	static void hadamardProduct(Matrix* A, Matrix* B, Matrix* R);
	static void minus(Matrix* A, Matrix* B, Matrix* R);

private:
	static cublasHandle_t cublasHandle;
	static cudnnHandle_t cudnnHandle;
	static cublasOperation_t get_cublas_trans(MatrixTransType t) { return t == mt_NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; }
	static CBLAS_TRANSPOSE get_cblas_trans(MatrixTransType t) { return t == mt_NoTrans ? CblasNoTrans : CblasTrans; }

	cudnnTensorDescriptor_t tensorDes = nullptr;

	static cudnnTensorDescriptor_t td;
	static cudnnActivationDescriptor_t ad;
	static cudnnOpTensorDescriptor_t od;
	static cudnnPoolingDescriptor_t pd;
	static cudnnConvolutionDescriptor_t cd;
	static cudnnFilterDescriptor_t fd;

	//������ԣ�
	double* mallocData(int size);
	void freeData();
	
	//�����������뽻����ԣ�
	double* malloc_getDataFromDevice();
	void freeDataForDevice(double* temp);
	double* mallocDataForDevice();
	void set_freeDataToDevice(double* temp);

public:
	static void setTensorDes(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w);

	static void poolingForward(ResampleType re, Matrix* X, Matrix* Y, 
		int window_w, int window_h, int stride_w, int stride_h, int* recordPos = nullptr);
	static void poolingBackward(ResampleType re, Matrix* Y, Matrix* dY, Matrix* X, Matrix* dX, 
		int window_w, int window_h, int stride_w, int stride_h, int* recordPos = nullptr);

	static void convolutionForward(Matrix* X, Matrix* conv_kernel, Matrix* Y,
		int m_subA, int n_subA, int m_subR, int n_subR, int countPerGroup);

	static void selectFunction(MatrixCudaType useCuda, double* x, double* y, int size,
		std::function<int(double*, double*, int)> f1, std::function<int(double*, double*, int)> f2);

	static void setActive(cudnnActivationMode_t am);
	static void activeForward(ActiveFunctionType af, Matrix* X, Matrix* Y);
	static void activeBackward(ActiveFunctionType af, Matrix* Y, Matrix* X, Matrix* dX);

};

