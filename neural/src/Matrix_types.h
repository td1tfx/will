#pragma once
#include "types.h"
#include "cublas_real.h"
#include "cblas_real.h"

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

struct CudaParameters
{
    friend class Matrix;
    bool inited = false;
    MatrixCudaType globalUseCuda = mc_NoCuda;
    cublasHandle_t cublasH;
    cudnnHandle_t cudnnH;

    Cublas* cublas;
    Cblas* cblas;

    //Ϊ���Ҹо������ܶ���
    //һ����˵����X��A��dX��dA�У���X�ļ�������Ϊ׼������ʱһ�㲻����������
    //��Щ�����ĳ�ֵ��Ϊ�գ�һ����˵���㷨ֻ����һ�Σ�֮���ظ�����
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

    //����һ���Դ���ΪһЩ���ܵĿռ�
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

