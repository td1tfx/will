#pragma once
#define varName(a) #a
typedef double DataType;

//���������
typedef enum
{
	af_Sigmoid = 0,
	af_Linear,
	af_Softmax,
	af_Tanh,
	af_Findmax,
	af_Softplus,
	af_ReLU,
} ActiveFunctionType;

//��������
typedef enum
{
	re_Max = 0,
	re_Average,
} ResampleType;

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
} CostFunctionType;
