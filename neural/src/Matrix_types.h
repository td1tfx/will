#pragma once
#include "types.h"
#include "cublas_real.h"
#include "cblas_real.h"

//数据位置（是否需要自己析构数据）
typedef enum
{
    md_Outside = 0,
    md_Inside,
} MatrixDataType;

//数据是否存储于CUDA设备
typedef enum
{
    mc_NoCuda = 0,
    mc_UseCuda,
} MatrixCudaType;

struct CudaParameters
{
    friend class Matrix;
    bool inited = false;
    MatrixCudaType globalUseCuda = mc_NoCuda;
    cublasHandle_t cublasH;
    cudnnHandle_t cudnnH;

    Cublas* cublas;
    Cblas* cblas;

    //为何我感觉这样很恶心
    //一般来说，在X，A，dX，dA中，以X的计算设置为准，反向时一般不再重新设置
    //这些参量的初值均为空，一般来说，算法只设置一次，之后不重复设置
    cudnnTensorDescriptor_t TensorDesc = nullptr;
    cudnnTensorDescriptor_t asTensorDesc = nullptr;
    cudnnActivationDescriptor_t ActivationDesc = nullptr;
    cudnnOpTensorDescriptor_t OpTensorDesc = nullptr;
    cudnnPoolingDescriptor_t PoolingDesc = nullptr;
    cudnnConvolutionDescriptor_t ConvolutionDesc = nullptr;
    cudnnFilterDescriptor_t FilterDesc = nullptr;
    cudnnRNNDescriptor_t RNNDesc = nullptr;
    cudnnDropoutDescriptor_t DropoutDesc = nullptr;
    cudnnSpatialTransformerDescriptor_t SpatialTransformerDesc = nullptr;
    cudnnLRNDescriptor_t LRNDesc = nullptr;

    //开辟一块显存作为一些功能的空间
    void* workspace;
    const int workspace_size = 1024 * 1024 * 128;
    void initWorkbase() { cudaMalloc(&workspace, workspace_size); }
private:
    CudaParameters()
    {
        cudnnCreateDescriptor(&TensorDesc);
        cudnnCreateDescriptor(&asTensorDesc);
        cudnnCreateDescriptor(&ActivationDesc);
        cudnnCreateDescriptor(&OpTensorDesc);
        cudnnCreateDescriptor(&PoolingDesc);
        cudnnCreateDescriptor(&ConvolutionDesc);
        cudnnCreateDescriptor(&FilterDesc);
        cudnnCreateDescriptor(&RNNDesc);
        cudnnCreateDescriptor(&DropoutDesc);
        cudnnCreateDescriptor(&SpatialTransformerDesc);
        cudnnCreateDescriptor(&LRNDesc);
    }

    ~CudaParameters() 
    {
        cudnnDestroyDescriptor(TensorDesc);
        cudnnDestroyDescriptor(asTensorDesc);
        cudnnDestroyDescriptor(ActivationDesc);
        cudnnDestroyDescriptor(OpTensorDesc);
        cudnnDestroyDescriptor(PoolingDesc);
        cudnnDestroyDescriptor(ConvolutionDesc);
        cudnnDestroyDescriptor(FilterDesc);
        cudnnDestroyDescriptor(RNNDesc);
        cudnnDestroyDescriptor(DropoutDesc);
        cudnnDestroyDescriptor(SpatialTransformerDesc);
        cudnnDestroyDescriptor(LRNDesc);
    }
};

struct MatrixConstant
{
    friend class Matrix;
    const real real_1 = 1;
    const real real_0 = 0;
private:
    MatrixConstant() {}
};

