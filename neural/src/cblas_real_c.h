#pragma once
#include "cblas.h"

float cblas_dot(const int N, const float* X, const int incX, const float* Y,
                const int incY);
double cblas_dot(const int N, const double* X, const int incX, const double* Y,
                 const int incY);
float cblas_nrm2(const int N, const float* X, const int incX);
float cblas_asum(const int N, const float* X, const int incX);
double cblas_nrm2(const int N, const double* X, const int incX);
double cblas_asum(const int N, const double* X, const int incX);
CBLAS_INDEX cblas_iamax(const int N, const float  *X, const int incX);
CBLAS_INDEX cblas_iamax(const int N, const double *X, const int incX);
void cblas_swap(const int N, float* X, const int incX, float* Y,
                const int incY);
void cblas_copy(const int N, const float* X, const int incX, float* Y,
                const int incY);
void cblas_axpy(const int N, const float alpha, const float* X, const int incX,
                float* Y, const int incY);
void cblas_swap(const int N, double* X, const int incX, double* Y,
                const int incY);
void cblas_copy(const int N, const double* X, const int incX, double* Y,
                const int incY);
void cblas_axpy(const int N, const double alpha, const double* X,
                const int incX, double* Y, const int incY);
void cblas_rotg(float* a, float* b, float* c, float* s);
void cblas_rotmg(float* d1, float* d2, float* b1, const float b2, float* P);
void cblas_rot(const int N, float* X, const int incX, float* Y, const int incY,
               const float c, const float s);
void cblas_rotm(const int N, float* X, const int incX, float* Y, const int incY,
                const float* P);
void cblas_rotg(double* a, double* b, double* c, double* s);
void cblas_rotmg(double* d1, double* d2, double* b1, const double b2,
                 double* P);
void cblas_rot(const int N, double* X, const int incX, double* Y,
               const int incY, const double c, const double s);
void cblas_rotm(const int N, double* X, const int incX, double* Y,
                const int incY, const double* P);
void cblas_scal(const int N, const float alpha, float *X, const int incX);
void cblas_scal(const int N, const double alpha, double *X, const int incX);
void cblas_gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N, const float alpha, const float* A,
                const int lda, const float* X, const int incX, const float beta, float* Y,
                const int incY);
void cblas_gbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N, const int KL, const int KU,
                const float alpha, const float* A, const int lda, const float* X,
                const int incX, const float beta, float* Y, const int incY);
void cblas_trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const float* A, const int lda, float* X, const int incX);
void cblas_tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const int K, const float* A, const int lda, float* X, const int incX);
void cblas_tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const float* Ap, float* X, const int incX);
void cblas_trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const float* A, const int lda, float* X, const int incX);
void cblas_tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const int K, const float* A, const int lda, float* X, const int incX);
void cblas_tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const float* Ap, float* X, const int incX);
void cblas_gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N, const double alpha, const double* A,
                const int lda, const double* X, const int incX, const double beta, double* Y,
                const int incY);
void cblas_gbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N, const int KL, const int KU,
                const double alpha, const double* A, const int lda, const double* X,
                const int incX, const double beta, double* Y, const int incY);
void cblas_trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const double* A, const int lda, double* X, const int incX);
void cblas_tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const int K, const double* A, const int lda, double* X, const int incX);
void cblas_tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const double* Ap, double* X, const int incX);
void cblas_trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const double* A, const int lda, double* X, const int incX);
void cblas_tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const int K, const double* A, const int lda, double* X, const int incX);
void cblas_tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N,
                const double* Ap, double* X, const int incX);
void cblas_symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float* A, const int lda,
                const float* X, const int incX, const float beta, float* Y, const int incY);
void cblas_sbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const int K, const float alpha, const float* A, const int lda,
                const float* X, const int incX, const float beta, float* Y, const int incY);
void cblas_spmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float* Ap, const float* X,
                const int incX, const float beta, float* Y, const int incY);
void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
               const float alpha, const float* X, const int incX, const float* Y,
               const int incY, float* A, const int lda);
void cblas_syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const float alpha, const float* X, const int incX, float* A,
               const int lda);
void cblas_spr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const float alpha, const float* X, const int incX, float* Ap);
void cblas_syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float* X, const int incX,
                const float* Y, const int incY, float* A, const int lda);
void cblas_spr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float* X, const int incX,
                const float* Y, const int incY, float* A);
void cblas_symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double* A, const int lda,
                const double* X, const int incX, const double beta, double* Y, const int incY);
void cblas_sbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const int K, const double alpha, const double* A,
                const int lda, const double* X, const int incX, const double beta, double* Y,
                const int incY);
void cblas_spmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double* Ap, const double* X,
                const int incX, const double beta, double* Y, const int incY);
void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
               const double alpha, const double* X, const int incX, const double* Y,
               const int incY, double* A, const int lda);
void cblas_syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const double alpha, const double* X, const int incX, double* A,
               const int lda);
void cblas_spr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const double alpha, const double* X, const int incX,
               double* Ap);
void cblas_syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double* X, const int incX,
                const double* Y, const int incY, double* A, const int lda);
void cblas_spr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double* X, const int incX,
                const double* Y, const int incY, double* A);
void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const float alpha, const float* A, const int lda, const float* B,
                const int ldb, const float beta, float* C, const int ldc);
void cblas_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                const enum CBLAS_UPLO Uplo, const int M, const int N, const float alpha,
                const float* A, const int lda, const float* B, const int ldb, const float beta,
                float* C, const int ldc);
void cblas_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                const float alpha, const float* A, const int lda, const float beta, float* C,
                const int ldc);
void cblas_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const float* A, const int lda, const float* B, const int ldb,
                 const float beta, float* C, const int ldc);
void cblas_trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha,
                const float* A, const int lda, float* B, const int ldb);
void cblas_trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha,
                const float* A, const int lda, float* B, const int ldb);
void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const double alpha, const double* A, const int lda,
                const double* B, const int ldb, const double beta, double* C, const int ldc);
void cblas_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                const enum CBLAS_UPLO Uplo, const int M, const int N, const double alpha,
                const double* A, const int lda, const double* B, const int ldb,
                const double beta, double* C, const int ldc);
void cblas_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                const double alpha, const double* A, const int lda, const double beta,
                double* C, const int ldc);
void cblas_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double* A, const int lda, const double* B,
                 const int ldb, const double beta, double* C, const int ldc);
void cblas_trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha,
                const double* A, const int lda, double* B, const int ldb);
void cblas_trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha,
                const double* A, const int lda, double* B, const int ldb);

void cblas_gemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                const float alpha, const float* A, const int lda, const float* X,
                const int incX, const float beta, float* Y, const int incY);
void cblas_gbmv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                const int KL, const int KU, const float alpha, const float* A,
                const int lda, const float* X, const int incX, const float beta, float* Y,
                const int incY);
void cblas_trmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const float* A, const int lda,
                float* X, const int incX);
void cblas_tbmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const int K, const float* A,
                const int lda, float* X, const int incX);
void cblas_tpmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const float* Ap, float* X,
                const int incX);
void cblas_trsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const float* A, const int lda,
                float* X, const int incX);
void cblas_tbsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const int K, const float* A,
                const int lda, float* X, const int incX);
void cblas_tpsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const float* Ap, float* X,
                const int incX);
void cblas_gemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                const double alpha, const double* A, const int lda, const double* X,
                const int incX, const double beta, double* Y, const int incY);
void cblas_gbmv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                const int KL, const int KU, const double alpha, const double* A,
                const int lda, const double* X, const int incX, const double beta, double* Y,
                const int incY);
void cblas_trmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const double* A,
                const int lda, double* X, const int incX);
void cblas_tbmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const int K, const double* A,
                const int lda, double* X, const int incX);
void cblas_tpmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const double* Ap, double* X,
                const int incX);
void cblas_trsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const double* A,
                const int lda, double* X, const int incX);
void cblas_tbsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const int K, const double* A,
                const int lda, double* X, const int incX);
void cblas_tpsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_DIAG Diag, const int N, const double* Ap, double* X,
                const int incX);
void cblas_symv(const enum CBLAS_UPLO Uplo, const int N, const float alpha,
                const float* A, const int lda, const float* X, const int incX,
                const float beta, float* Y, const int incY);
void cblas_sbmv(const enum CBLAS_UPLO Uplo, const int N, const int K,
                const float alpha, const float* A, const int lda, const float* X,
                const int incX, const float beta, float* Y, const int incY);
void cblas_spmv(const enum CBLAS_UPLO Uplo, const int N, const float alpha,
                const float* Ap, const float* X, const int incX, const float beta,
                float* Y, const int incY);
void cblas_ger(const int M, const int N, const float alpha, const float* X,
               const int incX, const float* Y, const int incY, float* A, const int lda);
void cblas_syr(const enum CBLAS_UPLO Uplo, const int N, const float alpha,
               const float* X, const int incX, float* A, const int lda);
void cblas_spr(const enum CBLAS_UPLO Uplo, const int N, const float alpha,
               const float* X, const int incX, float* Ap);
void cblas_syr2(const enum CBLAS_UPLO Uplo, const int N, const float alpha,
                const float* X, const int incX, const float* Y, const int incY, float* A,
                const int lda);
void cblas_spr2(const enum CBLAS_UPLO Uplo, const int N, const float alpha,
                const float* X, const int incX, const float* Y, const int incY, float* A);
void cblas_symv(const enum CBLAS_UPLO Uplo, const int N, const double alpha,
                const double* A, const int lda, const double* X, const int incX,
                const double beta, double* Y, const int incY);
void cblas_sbmv(const enum CBLAS_UPLO Uplo, const int N, const int K,
                const double alpha, const double* A, const int lda, const double* X,
                const int incX, const double beta, double* Y, const int incY);
void cblas_spmv(const enum CBLAS_UPLO Uplo, const int N, const double alpha,
                const double* Ap, const double* X, const int incX, const double beta,
                double* Y, const int incY);
void cblas_ger(const int M, const int N, const double alpha, const double* X,
               const int incX, const double* Y, const int incY, double* A,
               const int lda);
void cblas_syr(const enum CBLAS_UPLO Uplo, const int N, const double alpha,
               const double* X, const int incX, double* A, const int lda);
void cblas_spr(const enum CBLAS_UPLO Uplo, const int N, const double alpha,
               const double* X, const int incX, double* Ap);
void cblas_syr2(const enum CBLAS_UPLO Uplo, const int N, const double alpha,
                const double* X, const int incX, const double* Y, const int incY,
                double* A, const int lda);
void cblas_spr2(const enum CBLAS_UPLO Uplo, const int N, const double alpha,
                const double* X, const int incX, const double* Y, const int incY,
                double* A);
void cblas_gemm(const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                const float alpha,
                const float* A, const int lda, const float* B, const int ldb, const float beta,
                float* C, const int ldc);
void cblas_symm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const int M, const int N, const float alpha, const float* A, const int lda,
                const float* B, const int ldb, const float beta, float* C, const int ldc);
void cblas_syrk(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                const int N, const int K, const float alpha, const float* A,
                const int lda, const float beta, float* C, const int ldc);
void cblas_syr2k(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                 const int N, const int K, const float alpha, const float* A,
                 const int lda, const float* B, const int ldb, const float beta, float* C,
                 const int ldc);
void cblas_trmm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M,
                const int N, const float alpha, const float* A, const int lda, float* B,
                const int ldb);
void cblas_trsm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M,
                const int N, const float alpha, const float* A, const int lda, float* B,
                const int ldb);
void cblas_gemm(const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                const double alpha,
                const double* A, const int lda, const double* B, const int ldb,
                const double beta, double* C, const int ldc);
void cblas_symm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const int M, const int N, const double alpha, const double* A, const int lda,
                const double* B, const int ldb, const double beta, double* C, const int ldc);
void cblas_syrk(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                const int N, const int K, const double alpha, const double* A,
                const int lda, const double beta, double* C, const int ldc);
void cblas_syr2k(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                 const int N, const int K, const double alpha, const double* A,
                 const int lda, const double* B, const int ldb, const double beta, double* C,
                 const int ldc);
void cblas_trmm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M,
                const int N, const double alpha, const double* A, const int lda, double* B,
                const int ldb);
void cblas_trsm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M,
                const int N, const double alpha, const double* A, const int lda, double* B,
                const int ldb);

