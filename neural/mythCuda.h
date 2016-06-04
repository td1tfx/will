#pragma once

#ifdef _USE_CUDA
/* Includes, cuda */
#include "lib/cublas/cuda_runtime.h"
#include "lib/cublas/cublas_v2.h"
#include "lib/cublas/helper_cuda.h"
#endif

#include "lib/cblas.h"

class mythCuda
{
public:
	mythCuda();
	~mythCuda();

	
	static mythCuda* _mythcuda;
	static bool UseCublas;

#ifdef _USE_CUDA
	cublasHandle_t handle;
	static void init()
	{
		if (HasDevice())
		{
			_mythcuda = new mythCuda();
			UseCublas = true;
		}
		else
		{
			UseCublas = false;
		}
	}
#else
	static void init() { }
#endif

	static mythCuda* getInstance()
	{
		if (_mythcuda)
			return _mythcuda;
		else
			return new mythCuda();
	}

	static bool HasDevice();
	double myth_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
	void myth_dgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
	void myth_dgemv(const CBLAS_TRANSPOSE TransA, const int M, const int N, 
		const double alpha, const double  *A, const int lda, const double  *X, const int incX, const double beta, double  *Y, const int incY);
private:
	double* d_A;
	double* d_B;
	double* d_C;
	void bind(const double* A, int sizeA, const double* B, int sizeB, double* R, int sizeR);
};
