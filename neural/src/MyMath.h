#pragma once
#include <cmath>
#include <cstring>

namespace MyMath
{
#define MYMATH_VECTOR(fv, f) template<typename T> void fv(const T* x, T* a, int size) { for(int i=0;i<size;i++){a[i]=f(x[i]);} }
#define MYMATH_VECTOR_B(fv, content) template<typename T> void fv(const T* a, const T* da,const T* x, T* dx, int size) { for(int i=0;i<size;i++){dx[i]=(content);} }

	template<typename T> T sigmoid(T x) { return 1 / (1 + exp(-x)); }
	template<typename T> T softplus(T x) { return log(1 + exp(x)); }
	template<typename T> T relu(T x) { return x > 0 ? x : 0; }

	MYMATH_VECTOR(log_v, log);
	MYMATH_VECTOR(exp_v, exp);

	MYMATH_VECTOR(sigmoid_v, sigmoid);
	MYMATH_VECTOR(relu_v, relu);
	MYMATH_VECTOR(tanh_v, tanh);
	MYMATH_VECTOR(softplus_v, softplus);
	template<typename T> void linear_v(T* x, T* a, int size) { memcpy(a, x, sizeof(T)*size); }

	MYMATH_VECTOR_B(exp_vb, a[i]);
	MYMATH_VECTOR_B(sigmoid_vb, a[i] * (1 - a[i]) * da[i]); //sigmoid����ֱ��ʹ��a����
	MYMATH_VECTOR_B(relu_vb, x[i] > 0 ? 1 : 0);
	MYMATH_VECTOR_B(tanh_vb, (1 - a[i] * a[i]) * da[i]);
	MYMATH_VECTOR_B(softplus_vb, sigmoid(x[i]));
	MYMATH_VECTOR_B(linear_vb, 1);

	template<typename T> int softmax_vb_sub(const T* a, const T* da, T v, T* dx, int size)
	{
		for (int i = 0; i < size; i++)
		{
			dx[i] = a[i] * (da[i] - v);
		}
		return 0;
	}

	template<typename T> int softmaxloss_vb_sub(const T* a, const T* da, T v, T* dx, int size)
	{
		for (int i = 0; i < size; i++)
		{
			dx[i] = da[i] - v*exp(a[i]);
		}
		return 0;
	}

	template<typename T> bool inbox(T _x, T _y, T x, T y, T w, T h)
	{
		return _x >= x && _y >= y && _x < x + h && _y < y + h;
	}

#undef MYMATH_VECTOR
#undef MYMATH_VECTOR_B
};

