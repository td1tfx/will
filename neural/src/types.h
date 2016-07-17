#pragma once
//����һЩ���Ͷ��壬���õĺ������ģ��

#define _USE_CUDA
//#define _DOUBLE_PRECISION

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include "cudnn.h"

#ifdef _USE_CUDA
#pragma comment (lib, "cudart.lib")
#pragma comment (lib, "cublas.lib")
#pragma comment (lib, "cudnn.lib")
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

//Ĭ��ʹ�õ����ȣ���ʱӦʹ��real���帡����
#ifndef _DOUBLE_PRECISION
#define _SINGLE_PRECISION 
typedef float real;
#else 
typedef double real;
#endif


//���������
typedef enum
{
    af_Sigmoid     = CUDNN_ACTIVATION_SIGMOID,
    af_ReLU        = CUDNN_ACTIVATION_RELU,
    af_Tanh        = CUDNN_ACTIVATION_TANH,
    af_ClippedReLU = CUDNN_ACTIVATION_CLIPPED_RELU,
    af_Softmax,
    af_SoftmaxLoss,
    af_Linear,
    af_Findmax,
    af_Softplus,
    af_Dropout,
    af_DivisiveNormalization,
    af_BatchNormalization,
    af_SpatialTransformer,
} ActiveFunctionType;

//�������࣬��cuDNNֱ�Ӷ�Ӧ����������ת��
typedef enum
{
    pl_Max               = CUDNN_POOLING_MAX,
    pl_Average_Padding   = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    pl_Average_NoPadding = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
} PoolingType;

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
    cf_LogLikelihood,
} CostFunctionType;


#ifdef _SINGLE_PRECISION
#define REAL_MAX FLT_MAX
#define MYCUDNN_DATA_REAL CUDNN_DATA_FLOAT
#else
#define REAL_MAX DBL_MAX
#define MYCUDNN_DATA_REAL CUDNN_DATA_DOUBLE
#endif 

#ifndef _USE_CUDA
#define CUBLAS_FUNC(func) 
#define CUBLAS_FUNC_I(func) 
#endif


