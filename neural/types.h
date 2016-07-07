#pragma once

#define _USE_CUDA

#include "lib/cuda/cuda_runtime.h"
#include "lib/cuda/cublas_v2.h"
#include "lib/cuda/helper_cuda.h"
#include "lib/cuda/cudnn.h"

#ifdef _USE_CUDA
#pragma comment (lib, "cudart.lib")
#pragma comment (lib, "cublas.lib")
#pragma comment (lib, "cudnn.lib")

/*
#pragma comment (lib, "neural-cuda.lib")

#ifdef __cplusplus
#define HBAPI extern "C" __declspec (dllimport)
#else
#define HBAPI __declspec (dllimport)
#endif

int _stdcall cuda_hadamardProduct(const double *A, const double *B, double *R, unsigned int size);
int _stdcall cuda_sigmoid(double *A, double *B, unsigned int size);
int _stdcall cuda_dsigmoid(double *A, double *B, unsigned int size);
int _stdcall cuda_exp(double *A, double *B, unsigned int size);
*/
#else
//��cuda����Ч��ʱ����������ʹ�ù���cuda����
//���������֪���ò���

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

#define cudnnSetTensor4dDescriptor
#define cudnnSetPooling2dDescriptor
#define cudnnSetOpTensorDescriptor
#define cudnnSetActivationDescriptor
#define cudnnCreateTensorDescriptor
#define cudnnDestroyTensorDescriptor

#define cudnnSetTensor
#define cudnnOpTensor
#define cudnnPoolingForward
#define cudnnPoolingBackward
#define cudnnSoftmaxForward
#define cudnnActivationForward
#define cudnnActivationBackward

#endif

#define varName(a) #a

#ifndef _DOUBLE_PRECISION
#define _SINGLE_PRECISION 
typedef float real;
#else 
typedef double real;
#endif


//���������
typedef enum
{
	af_Sigmoid = 0,
	af_Linear,
	af_Softmax,
	af_Tanh,
	af_Findmax,
	af_Softplus,
	af_ReLU,
} ActiveFunctionType;

//�������࣬��cuDNNֱ�Ӷ�Ӧ����������ת��
typedef enum
{
	re_Max = CUDNN_POOLING_MAX,
	re_Average_Padding = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
	re_Average_NoPadding = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
} ResampleType;

//������ࣨ��������⣬������
typedef enum
{
	cv_1toN = 0,
	cv_NtoN,
} ConvolutionType;

//���ۺ�������
typedef enum
{
	cf_RMSE,
	cf_CrossEntropy,
} CostFunctionType;


#ifdef _SINGLE_PRECISION
#define CUDNN_DATA_real CUDNN_DATA_FLOAT  //��Сд��ͬ������
#define CBLAS_FUNC(func) cblas_s##func
#define CBLAS_FUNC_I(func) cblas_is##func
#define CUBLAS_FUNC(func) cublasS##func
#define CUBLAS_FUNC_I(func) cublasIs##func
#else
#define CUDNN_DATA_real CUDNN_DATA_DOUBLE
#define CBLAS_FUNC(func) cblas_d##func
#define CBLAS_FUNC_I(func) cblas_id##func
#define CUBLAS_FUNC(func) cublasD##func
#define CUBLAS_FUNC_I(func) cublasId##func
#endif 

#ifndef _USE_CUDA
#define CUBLAS_FUNC(func) 
#define CUBLAS_FUNC_I(func) 
#endif