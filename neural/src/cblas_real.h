#pragma once
#include "cblas.h"

class Cblas
{
public:
    template<typename T>
    inline static T cblas_dot(const int N, const T* X, const int incX, const T* Y, const int incY);

    template<typename T>
    inline static T cblas_nrm2(const int N, const T* X, const int incX);

    template<typename T>
    inline static T cblas_asum(const int N, const T* X, const int incX);

    template<typename T>
    inline static CBLAS_INDEX cblas_iamax(const int N, const T* X, const int incX);

    template<typename T>
    inline static void cblas_swap(const int N, T* X, const int incX, T* Y, const int incY);

    template<typename T>
    inline static void cblas_copy(const int N, const T* X, const int incX, T* Y, const int incY);

    template<typename T>
    inline static void cblas_axpy(const int N, const T alpha, const T* X, const int incX, T* Y, const int incY);

    template<typename T>
    inline static void cblas_rotg(T* a, T* b, T* c, T* s);

    template<typename T>
    inline static void cblas_rotmg(T* d1, T* d2, T* b1, const T b2, T* P);

    template<typename T>
    inline static void cblas_rot(const int N, T* X, const int incX, T* Y, const int incY, const T c, const T s);

    template<typename T>
    inline static void cblas_rotm(const int N, T* X, const int incX, T* Y, const int incY, const T* P);

    template<typename T>
    inline static void cblas_scal(const int N, const T alpha, T* X, const int incX);

    template<typename T>
    inline static void cblas_gemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_gbmv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_trmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* Ap, T* X, const int incX);

    template<typename T>
    inline static void cblas_trsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* Ap, T* X, const int incX);

    template<typename T>
    inline static void cblas_symv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_sbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int K, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_spmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* Ap, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_ger(const enum CBLAS_ORDER Order, const int M, const int N, const T alpha, const T* X, const int incX, const T* Y, const int incY, T* A, const int lda);

    template<typename T>
    inline static void cblas_syr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, T* A, const int lda);

    template<typename T>
    inline static void cblas_spr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, T* Ap);

    template<typename T>
    inline static void cblas_syr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, const T* Y, const int incY, T* A, const int lda);

    template<typename T>
    inline static void cblas_spr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, const T* Y, const int incY, T* A);

    template<typename T>
    inline static void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const T alpha, const T* A, const int lda, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const T alpha, const T* A, const int lda, T* B, const int ldb);

    template<typename T>
    inline static void cblas_trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const T alpha, const T* A, const int lda, T* B, const int ldb);


    template<typename T>
    inline static void cblas_gemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_gbmv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_trmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tbmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tpmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* Ap, T* X, const int incX);

    template<typename T>
    inline static void cblas_trsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tbsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const T* A, const int lda, T* X, const int incX);

    template<typename T>
    inline static void cblas_tpsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const T* Ap, T* X, const int incX);

    template<typename T>
    inline static void cblas_symv(const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_sbmv(const enum CBLAS_UPLO Uplo, const int N, const int K, const T alpha, const T* A, const int lda, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_spmv(const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* Ap, const T* X, const int incX, const T beta, T* Y, const int incY);

    template<typename T>
    inline static void cblas_ger(const int M, const int N, const T alpha, const T* X, const int incX, const T* Y, const int incY, T* A, const int lda);

    template<typename T>
    inline static void cblas_syr(const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, T* A, const int lda);

    template<typename T>
    inline static void cblas_spr(const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, T* Ap);

    template<typename T>
    inline static void cblas_syr2(const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, const T* Y, const int incY, T* A, const int lda);

    template<typename T>
    inline static void cblas_spr2(const enum CBLAS_UPLO Uplo, const int N, const T alpha, const T* X, const int incX, const T* Y, const int incY, T* A);

    template<typename T>
    inline static void cblas_gemm(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_symm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_syrk(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const T alpha, const T* A, const int lda, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_syr2k(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc);

    template<typename T>
    inline static void cblas_trmm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const T alpha, const T* A, const int lda, T* B, const int ldb);

    template<typename T>
    inline static void cblas_trsm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const T alpha, const T* A, const int lda, T* B, const int ldb);

};
