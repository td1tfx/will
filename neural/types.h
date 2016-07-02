#pragma once
#define varName(a) #a
typedef double DataType;

//激活函数种类
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

//采样种类
typedef enum
{
	re_Max = 0,
	re_Average,
} ResampleType;

//卷积种类（这个有问题，待定）
typedef enum
{
	cv_1toN = 0,
	cv_NtoN,
} ConvolutionType;

//代价函数种类
typedef enum
{
	cf_RMSE,
	cf_CrossEntropy,
} CostFunctionType;
