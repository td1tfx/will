#pragma once
#include "cublas_v2.h"
#include "blas_types.h"

class Cublas : Blas
{
#ifdef VIRTUAL_BLAS
public:
#else
private:
#endif
    Cublas() {}
    ~Cublas() {}
private:
#ifndef VIRTUAL_BLAS
    static
#endif
    cublasHandle_t handle;
    BLAS_FUNC cublasOperation_t get_trans(MatrixTransType t)
    { return t == Matrix_NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; }
    BLAS_FUNC cublasFillMode_t get_uplo(MatrixFillType t)
    { return t == Matrix_Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER; }
    BLAS_FUNC cublasDiagType_t get_diag(MatrixDiagType t)
    { return t == Matrix_NonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT; }
    BLAS_FUNC cublasSideMode_t get_side(MatrixSideType t)
    { return t == Matrix_Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; }
public:
    static cublasStatus_t init()
    { return cublasCreate(&handle); }
    static void destroy()
    { cublasDestroy(handle); }
    void set_handle(cublasHandle_t h) { handle = h; }
public:
    BLAS_FUNC float dot(const int N, const float* X, const int incX, const float* Y, const int incY)
    { float r; cublasSdot(handle, N, X, incX, Y, incY, &r); return r; }
    BLAS_FUNC double dot(const int N, const double* X, const int incX, const double* Y, const int incY)
    { double r; cublasDdot(handle, N, X, incX, Y, incY, &r); return r; }
    BLAS_FUNC float nrm2(const int N, const float* X, const int incX)
    { float r; cublasSnrm2(handle, N, X, incX, &r); return r; }
    BLAS_FUNC float asum(const int N, const float* X, const int incX)
    { float r; cublasSasum(handle, N, X, incX, &r); return r; }
    BLAS_FUNC double nrm2(const int N, const double* X, const int incX)
    { double r; cublasDnrm2(handle, N, X, incX, &r); return r; }
    BLAS_FUNC double asum(const int N, const double* X, const int incX)
    { double r; cublasDasum(handle, N, X, incX, &r); return r; }
    BLAS_FUNC int iamax(const int N, const float* X, const int incX)
    { int r; cublasIsamax(handle, N, X, incX, &r); return r - 1; }
    BLAS_FUNC int iamax(const int N, const double* X, const int incX)
    { int r; cublasIdamax(handle, N, X, incX, &r); return r - 1; }
    BLAS_FUNC void swap(const int N, float* X, const int incX, float* Y, const int incY)
    { cublasSswap(handle, N, X, incX, Y, incY); }
    BLAS_FUNC void copy(const int N, const float* X, const int incX, float* Y, const int incY)
    { cublasScopy(handle, N, X, incX, Y, incY); }
    BLAS_FUNC void axpy(const int N, const float alpha, const float* X, const int incX, float* Y, const int incY)
    { cublasSaxpy(handle, N, &alpha, X, incX, Y, incY); }
    BLAS_FUNC void swap(const int N, double* X, const int incX, double* Y, const int incY)
    { cublasDswap(handle, N, X, incX, Y, incY); }
    BLAS_FUNC void copy(const int N, const double* X, const int incX, double* Y, const int incY)
    { cublasDcopy(handle, N, X, incX, Y, incY); }
    BLAS_FUNC void axpy(const int N, const double alpha, const double* X, const int incX, double* Y, const int incY)
    { cublasDaxpy(handle, N, &alpha, X, incX, Y, incY); }
    BLAS_FUNC void rotg(float* a, float* b, float* c, float* s)
    { cublasSrotg(handle, a, b, c, s); }
    BLAS_FUNC void rotmg(float* d1, float* d2, float* b1, const float b2, float* P)
    { cublasSrotmg(handle, d1, d2, b1, &b2, P); }
    BLAS_FUNC void rot(const int N, float* X, const int incX, float* Y, const int incY, const float c, const float s)
    { cublasSrot(handle, N, X, incX, Y, incY, &c, &s); }
    BLAS_FUNC void rotm(const int N, float* X, const int incX, float* Y, const int incY, const float* P)
    { cublasSrotm(handle, N, X, incX, Y, incY, P); }
    BLAS_FUNC void rotg(double* a, double* b, double* c, double* s)
    { cublasDrotg(handle, a, b, c, s); }
    BLAS_FUNC void rotmg(double* d1, double* d2, double* b1, const double b2, double* P)
    { cublasDrotmg(handle, d1, d2, b1, &b2, P); }
    BLAS_FUNC void rot(const int N, double* X, const int incX, double* Y, const int incY, const double c, const double s)
    { cublasDrot(handle, N, X, incX, Y, incY, &c, &s); }
    BLAS_FUNC void rotm(const int N, double* X, const int incX, double* Y, const int incY, const double* P)
    { cublasDrotm(handle, N, X, incX, Y, incY, P); }
    BLAS_FUNC void scal(const int N, const float alpha, float* X, const int incX)
    { cublasSscal(handle, N, &alpha, X, incX); }
    BLAS_FUNC void scal(const int N, const double alpha, double* X, const int incX)
    { cublasDscal(handle, N, &alpha, X, incX); }
    BLAS_FUNC void gemv(const MatrixTransType TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    { cublasSgemv(handle, get_trans(TransA), M, N, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    { cublasSgbmv(handle, get_trans(TransA), M, N, KL, KU, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    { cublasStrmv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX); }
    BLAS_FUNC void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    { cublasStbmv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX); }
    BLAS_FUNC void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    { cublasStpmv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX); }
    BLAS_FUNC void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    { cublasStrsv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX); }
    BLAS_FUNC void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    { cublasStbsv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX); }
    BLAS_FUNC void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    { cublasStpsv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX); }
    BLAS_FUNC void gemv(const MatrixTransType TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    { cublasDgemv(handle, get_trans(TransA), M, N, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    { cublasDgbmv(handle, get_trans(TransA), M, N, KL, KU, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    { cublasDtrmv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX); }
    BLAS_FUNC void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    { cublasDtbmv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX); }
    BLAS_FUNC void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    { cublasDtpmv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX); }
    BLAS_FUNC void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    { cublasDtrsv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX); }
    BLAS_FUNC void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    { cublasDtbsv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX); }
    BLAS_FUNC void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    { cublasDtpsv(handle, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX); }
    BLAS_FUNC void symv(const MatrixFillType Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    { cublasSsymv(handle, get_uplo(Uplo), N, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void sbmv(const MatrixFillType Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    { cublasSsbmv(handle, get_uplo(Uplo), N, K, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void spmv(const MatrixFillType Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY)
    { cublasSspmv(handle, get_uplo(Uplo), N, &alpha, Ap, X, incX, &beta, Y, incY); }
    BLAS_FUNC void ger(const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    { cublasSger(handle, M, N, &alpha, X, incX, Y, incY, A, lda); }
    BLAS_FUNC void syr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda)
    { cublasSsyr(handle, get_uplo(Uplo), N, &alpha, X, incX, A, lda); }
    BLAS_FUNC void spr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap)
    { cublasSspr(handle, get_uplo(Uplo), N, &alpha, X, incX, Ap); }
    BLAS_FUNC void syr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    { cublasSsyr2(handle, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A, lda); }
    BLAS_FUNC void spr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A)
    { cublasSspr2(handle, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A); }
    BLAS_FUNC void symv(const MatrixFillType Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    { cublasDsymv(handle, get_uplo(Uplo), N, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void sbmv(const MatrixFillType Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    { cublasDsbmv(handle, get_uplo(Uplo), N, K, &alpha, A, lda, X, incX, &beta, Y, incY); }
    BLAS_FUNC void spmv(const MatrixFillType Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY)
    { cublasDspmv(handle, get_uplo(Uplo), N, &alpha, Ap, X, incX, &beta, Y, incY); }
    BLAS_FUNC void ger(const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    { cublasDger(handle, M, N, &alpha, X, incX, Y, incY, A, lda); }
    BLAS_FUNC void syr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda)
    { cublasDsyr(handle, get_uplo(Uplo), N, &alpha, X, incX, A, lda); }
    BLAS_FUNC void spr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap)
    { cublasDspr(handle, get_uplo(Uplo), N, &alpha, X, incX, Ap); }
    BLAS_FUNC void syr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    { cublasDsyr2(handle, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A, lda); }
    BLAS_FUNC void spr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A)
    { cublasDspr2(handle, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A); }
    BLAS_FUNC void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    { cublasSgemm(handle, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc); }
    BLAS_FUNC void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    { cublasSsymm(handle, get_side(Side), get_uplo(Uplo), M, N, &alpha, A, lda, B, ldb, &beta, C, ldc); }
    BLAS_FUNC void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc)
    { cublasSsyrk(handle, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, &beta, C, ldc); }
    BLAS_FUNC void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    { cublasSsyr2k(handle, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, B, ldb, &beta, C, ldc); }
    BLAS_FUNC void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    { cublasStrmm(handle, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb, B, ldb); }
    BLAS_FUNC void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    { cublasStrsm(handle, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb); }
    BLAS_FUNC void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    { cublasDgemm(handle, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc); }
    BLAS_FUNC void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    { cublasDsymm(handle, get_side(Side), get_uplo(Uplo), M, N, &alpha, A, lda, B, ldb, &beta, C, ldc); }
    BLAS_FUNC void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc)
    { cublasDsyrk(handle, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, &beta, C, ldc); }
    BLAS_FUNC void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    { cublasDsyr2k(handle, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, B, ldb, &beta, C, ldc); }
    BLAS_FUNC void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    { cublasDtrmm(handle, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb, B, ldb); }
    BLAS_FUNC void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    { cublasDtrsm(handle, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb); }
};


