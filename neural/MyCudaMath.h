#pragma once

#ifdef _MSC_VER
#define _USE_CUDA
#endif

#ifdef _USE_CUDA
#include "lib/cuda/cuda_runtime.h"
#include "lib/cuda/cublas_v2.h"
#include "lib/cuda/helper_cuda.h"
#include "lib/cuda/cudnn.h"

#pragma comment (lib, "cublas.lib")
#pragma comment (lib, "cudart_static.lib")
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

//在cuda不生效的时候，屏蔽所有使用过的cuda函数
//这个方法不知道好不好

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

#define cudnnHandle_t int
#define cudnnTensorDescriptor_t int
#define cudnnActivationDescriptor_t int
#define cudnnTensorDescriptor_t int;
#define cudnnActivationDescriptor_t int;
#define cudnnOpTensorDescriptor_t int;
#define cudnnPoolingDescriptor_t int;
#define cudnnConvolutionDescriptor_t int;
#define cudnnFilterDescriptor_t int;

#endif
