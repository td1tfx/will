#include "cblas_real.h"

template<>
float Cblas::cblas_dot(const int N, const float* X, const int incX, const float* Y, const int incY)
{
    return cblas_sdot(N, X, incX, Y, incY);
}

template<>
double Cblas::cblas_dot(const int N, const double* X, const int incX, const double* Y, const int incY)
{
    return cblas_ddot(N, X, incX, Y, incY);
}

template<>
float Cblas::cblas_nrm2(const int N, const float* X, const int incX)
{
    return cblas_snrm2(N, X, incX);
}

template<>
float Cblas::cblas_asum(const int N, const float* X, const int incX)
{
    return cblas_sasum(N, X, incX);
}

template<>
double Cblas::cblas_nrm2(const int N, const double* X, const int incX)
{
    return cblas_dnrm2(N, X, incX);
}

template<>
double Cblas::cblas_asum(const int N, const double* X, const int incX)
{
    return cblas_dasum(N, X, incX);
}

template<>
CBLAS_INDEX Cblas::cblas_iamax(const int N, const float  *X, const int incX)
{
    return cblas_isamax(N, X, incX);
}

template<>
CBLAS_INDEX Cblas::cblas_iamax(const int N, const double *X, const int incX)
{
    return cblas_idamax(N, X, incX);
}

template<>
void Cblas::cblas_swap(const int N, float* X, const int incX, float* Y, const int incY)
{
    cblas_sswap(N, X, incX, Y, incY);
}

template<>
void Cblas::cblas_copy(const int N, const float* X, const int incX, float* Y, const int incY)
{
    cblas_scopy(N, X, incX, Y, incY);
}

template<>
void Cblas::cblas_axpy(const int N, const float alpha, const float* X, const int incX, float* Y, const int incY)
{
    cblas_saxpy(N, alpha, X, incX, Y, incY);
}

template<>
void Cblas::cblas_swap(const int N, double* X, const int incX, double* Y, const int incY)
{
    cblas_dswap(N, X, incX, Y, incY);
}

template<>
void Cblas::cblas_copy(const int N, const double* X, const int incX, double* Y, const int incY)
{
    cblas_dcopy(N, X, incX, Y, incY);
}

template<>
void Cblas::cblas_axpy(const int N, const double alpha, const double* X, const int incX, double* Y, const int incY)
{
    cblas_daxpy(N, alpha, X, incX, Y, incY);
}

template<>
void Cblas::cblas_rotg(float* a, float* b, float* c, float* s)
{
    cblas_srotg(a, b, c, s);
}

template<>
void Cblas::cblas_rotmg(float* d1, float* d2, float* b1, const float b2, float* P)
{
    cblas_srotmg(d1, d2, b1, b2, P);
}

template<>
void Cblas::cblas_rot(const int N, float* X, const int incX, float* Y, const int incY, const float c, const float s)
{
    cblas_srot(N, X, incX, Y, incY, c, s);
}

template<>
void Cblas::cblas_rotm(const int N, float* X, const int incX, float* Y, const int incY, const float* P)
{
    cblas_srotm(N, X, incX, Y, incY, P);
}

template<>
void Cblas::cblas_rotg(double* a, double* b, double* c, double* s)
{
    cblas_drotg(a, b, c, s);
}

template<>
void Cblas::cblas_rotmg(double* d1, double* d2, double* b1, const double b2, double* P)
{
    cblas_drotmg(d1, d2, b1, b2, P);
}

template<>
void Cblas::cblas_rot(const int N, double* X, const int incX, double* Y, const int incY, const double c, const double s)
{
    cblas_drot(N, X, incX, Y, incY, c, s);
}

template<>
void Cblas::cblas_rotm(const int N, double* X, const int incX, double* Y, const int incY, const double* P)
{
    cblas_drotm(N, X, incX, Y, incY, P);
}

template<>
void Cblas::cblas_scal(const int N, const float alpha, float* X, const int incX)
{
    cblas_sscal(N, alpha, X, incX);
}

template<>
void Cblas::cblas_scal(const int N, const double alpha, double* X, const int incX)
{
    cblas_dscal(N, alpha, X, incX);
}

template<>
void Cblas::cblas_gemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_gbmv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_sgbmv(Order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_trmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* A, const int lda, float* X, const int incX)
{
    cblas_strmv(Order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
{
    cblas_stbmv(Order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* Ap, float* X, const int incX)
{
    cblas_stpmv(Order, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_trsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* A, const int lda, float* X, const int incX)
{
    cblas_strsv(Order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
{
    cblas_stbsv(Order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* Ap, float* X, const int incX)
{
    cblas_stpsv(Order, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_gemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_gbmv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dgbmv(Order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_trmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtrmv(Order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtbmv(Order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* Ap, double* X, const int incX)
{
    cblas_dtpmv(Order, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_trsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtrsv(Order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtbsv(Order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* Ap, double* X, const int incX)
{
    cblas_dtpsv(Order, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_symv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_ssymv(Order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_sbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_ssbmv(Order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_spmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_sspmv(Order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_ger(const enum CBLAS_ORDER Order, const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
{
    cblas_sger(Order, M, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_syr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda)
{
    cblas_ssyr(Order, Uplo, N, alpha, X, incX, A, lda);
}

template<>
void Cblas::cblas_spr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap)
{
    cblas_sspr(Order, Uplo, N, alpha, X, incX, Ap);
}

template<>
void Cblas::cblas_syr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
{
    cblas_ssyr2(Order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_spr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A)
{
    cblas_sspr2(Order, Uplo, N, alpha, X, incX, Y, incY, A);
}

template<>
void Cblas::cblas_symv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dsymv(Order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_sbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dsbmv(Order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_spmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dspmv(Order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_ger(const enum CBLAS_ORDER Order, const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
{
    cblas_dger(Order, M, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_syr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda)
{
    cblas_dsyr(Order, Uplo, N, alpha, X, incX, A, lda);
}

template<>
void Cblas::cblas_spr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap)
{
    cblas_dspr(Order, Uplo, N, alpha, X, incX, Ap);
}

template<>
void Cblas::cblas_syr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
{
    cblas_dsyr2(Order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_spr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A)
{
    cblas_dspr2(Order, Uplo, N, alpha, X, incX, Y, incY, A);
}

template<>
void Cblas::cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
    cblas_ssymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc)
{
    cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<>
void Cblas::cblas_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
    cblas_ssyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
{
    cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
{
    cblas_strsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
    cblas_dsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc)
{
    cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<>
void Cblas::cblas_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
    cblas_dsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
{
    cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
{
    cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_gemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_sgemv(CblasColMajor, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_gbmv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_sgbmv(CblasColMajor, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_trmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* A, const int lda, float* X, const int incX)
{
    cblas_strmv(CblasColMajor, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
{
    cblas_stbmv(CblasColMajor, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* Ap, float* X, const int incX)
{
    cblas_stpmv(CblasColMajor, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_trsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* A, const int lda, float* X, const int incX)
{
    cblas_strsv(CblasColMajor, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
{
    cblas_stbsv(CblasColMajor, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float* Ap, float* X, const int incX)
{
    cblas_stpsv(CblasColMajor, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_gemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dgemv(CblasColMajor, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_gbmv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dgbmv(CblasColMajor, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_trmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtrmv(CblasColMajor, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtbmv(CblasColMajor, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpmv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* Ap, double* X, const int incX)
{
    cblas_dtpmv(CblasColMajor, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_trsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtrsv(CblasColMajor, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template<>
void Cblas::cblas_tbsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtbsv(CblasColMajor, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template<>
void Cblas::cblas_tpsv(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double* Ap, double* X, const int incX)
{
    cblas_dtpsv(CblasColMajor, Uplo, TransA, Diag, N, Ap, X, incX);
}

template<>
void Cblas::cblas_symv(const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_ssymv(CblasColMajor, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_sbmv(const enum CBLAS_UPLO Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_ssbmv(CblasColMajor, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_spmv(const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY)
{
    cblas_sspmv(CblasColMajor, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_ger(const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
{
    cblas_sger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_syr(const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda)
{
    cblas_ssyr(CblasColMajor, Uplo, N, alpha, X, incX, A, lda);
}

template<>
void Cblas::cblas_spr(const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap)
{
    cblas_sspr(CblasColMajor, Uplo, N, alpha, X, incX, Ap);
}

template<>
void Cblas::cblas_syr2(const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
{
    cblas_ssyr2(CblasColMajor, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_spr2(const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A)
{
    cblas_sspr2(CblasColMajor, Uplo, N, alpha, X, incX, Y, incY, A);
}

template<>
void Cblas::cblas_symv(const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dsymv(CblasColMajor, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_sbmv(const enum CBLAS_UPLO Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dsbmv(CblasColMajor, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_spmv(const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY)
{
    cblas_dspmv(CblasColMajor, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

template<>
void Cblas::cblas_ger(const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
{
    cblas_dger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_syr(const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda)
{
    cblas_dsyr(CblasColMajor, Uplo, N, alpha, X, incX, A, lda);
}

template<>
void Cblas::cblas_spr(const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap)
{
    cblas_dspr(CblasColMajor, Uplo, N, alpha, X, incX, Ap);
}

template<>
void Cblas::cblas_syr2(const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
{
    cblas_dsyr2(CblasColMajor, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

template<>
void Cblas::cblas_spr2(const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A)
{
    cblas_dspr2(CblasColMajor, Uplo, N, alpha, X, incX, Y, incY, A);
}

template<>
void Cblas::cblas_gemm(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
    cblas_sgemm(CblasColMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_symm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
    cblas_ssymm(CblasColMajor, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_syrk(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc)
{
    cblas_ssyrk(CblasColMajor, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<>
void Cblas::cblas_syr2k(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
    cblas_ssyr2k(CblasColMajor, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_trmm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
{
    cblas_strmm(CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_trsm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
{
    cblas_strsm(CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_gemm(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
    cblas_dgemm(CblasColMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_symm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
    cblas_dsymm(CblasColMajor, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_syrk(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc)
{
    cblas_dsyrk(CblasColMajor, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<>
void Cblas::cblas_syr2k(const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
    cblas_dsyr2k(CblasColMajor, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void Cblas::cblas_trmm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
{
    cblas_dtrmm(CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

template<>
void Cblas::cblas_trsm(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
{
    cblas_dtrsm(CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}


