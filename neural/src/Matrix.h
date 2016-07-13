#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <cfloat>
#include "cblas.h"
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

//���������
//�������������Ƿ�װcuda����
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

	//һ�е�������Ϊһ����һ��ͼ�񣬾�������������
	//������ͼƬ����������������Ĭ���������ȣ�Ҳ����˵ͼƬ�;����˿�����Ϊ��ת�ñ���ģ���
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
	int whcn2i(int w, int h, int c, int n) { return w + h*W + c*W*H + n*C*W*H; }

	//����4������ע������������Դ��У�һ����˵���޷���ֵ�������
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

	//��������ָ�룬����������ܲ���ȫ�����ã���
	void resetDataPointer(real* d) { data = d; }

	//ʹ�������������Ҫ��Ϊ������ʱͬʱɾ������ָ�룬�����������ڸ�ɶ��
	void setInsideData(MatrixDataType id) { insideData = id; }

	real& operator [] (int i) { return data[i]; }

	static void initCuda();
	static void destroyCuda();

	void print(FILE* fout = stdout);
	int load(real* v, int n);
	void printAsVector(FILE* fout = stdout);
	int loadAsVector(real* v, int n);

	void memcpyDataIn(real* src, int size);
	void memcpyDataOut(real* dst, int size);
	void expand();
	int indexColMaxAbs(int c);
	real sumAbs();
	real sumColAbs(int c);
	real ddot();

	void initData(real v);
	void initRandom();
	void initInt(int a = 0);
	void multiply(real v);
	void colMultiply(real v, int c);

	static void cpyData(Matrix* dst, Matrix* src);
	void tryUploadToCuda();
	void tryDownloadFromCuda();
	void shareData(Matrix* A, int m, int n);

	//Ϊ��̬�����ڽ������ʹ���Դ�ʱ�͵���cuda�������㣬���ǵ�����Ӧ��֤���о���һ��
	//��δ����ȫ��UseCuda����Ϊ��ʹ��cuda��ʱ��Ҳ�п��ܴ������ڴ��еľ���
	static MatrixCudaType selectUseCuda(Matrix* A1 = nullptr, Matrix* A2 = nullptr, Matrix* A3 = nullptr, Matrix* A4 = nullptr);

	static void product(Matrix* A, Matrix* B, Matrix* R,
		real a = 1, real c = 0, MatrixTransType ta = mt_NoTrans, MatrixTransType tb = mt_NoTrans);
	static void productVector(Matrix* A, Matrix* B, Matrix* R,
		real a = 1, real c = 0, MatrixTransType ta = mt_NoTrans);
	static void productVector2(Matrix* A, Matrix* B, Matrix* R,
		real a = 1, real c = 0, MatrixTransType ta = mt_NoTrans);
	static void hadamardProduct(Matrix* A, Matrix* B, Matrix* R);
	static void add(Matrix* A, real b, Matrix* B, Matrix* R);
	static real dot(Matrix* A, int cA, Matrix* B, int cB);

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
	static void* workspace;
	static const int workspace_size = 1024 * 1024 * 32;

	//������ԣ�
	real* mallocData(int size);
	void freeData();

	//�����������뽻����ԣ�
	real* malloc_getDataFromDevice();
	void freeDataForDevice(real* temp);
	real* mallocDataForDevice();
	void set_freeDataToDevice(real* temp);

public:
	static void setTensorDes(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w);

	static void poolingForward(ResampleType re, Matrix* X, Matrix* A,
		int window_w, int window_h, int stride_w, int stride_h, int* recordPos = nullptr);
	static void poolingBackward(ResampleType re, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
		int window_w, int window_h, int stride_w, int stride_h, int* recordPos = nullptr);

	static void convolutionForward(Matrix* X, Matrix* W, Matrix* A, int* recordX = nullptr, int* recordW = nullptr);
	static void convolutionBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* W, Matrix* dW, Matrix* dB);
	static void convolution_sub(Matrix* A, int cA, Matrix* B, int cB, Matrix* C, int cC, Matrix* R, int n, int plus);

	static void selectFunction(MatrixCudaType useCuda, real* x, real* y, int size,
		std::function<int(real*, real*, int)> f1, std::function<int(real*, real*, int)> f2);

	static void setActive(cudnnActivationMode_t am);
	static void activeForward(ActiveFunctionType af, Matrix* X, Matrix* A);
	static void activeBackward(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX);
};

typedef Matrix Tensor;
