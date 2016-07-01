#pragma once

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

//ÆÁ±ÎËùÓÐcudaº¯Êý
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
