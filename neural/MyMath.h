#pragma once
#include <cmath>
#include <cstring>

namespace MyMath
{
#define MYMATH_FOR(f) do{for(int i=0;i<size;i++){y[i]=f(x[i]);}}while(0)
#define MYMATH_VECTOR(fv, f) template<typename T> int fv(const T* x, T* y, int size) { MYMATH_FOR(f); return 0; }
#define MYMATH_FOR_B(df) do{for(int i=0;i<size;i++){dx[i]=df(x[i])*da[i];}}while(0)
#define MYMATH_VECTOR_B(fv, df) template<typename T> int fv(const T* a,const T* da,const T* x, T* dx, int size) { MYMATH_FOR_B(df); return 0; }

	template<typename T> T sigmoid(T x) { return 1 / (1 + exp(-x)); }
	template<typename T> T dsigmoid(T x) { T t = 1 + exp(-x); return (t - 1) / (t*t); }
	template<typename T> T dsigmoid2(T a) { return a*(1 - a); }
	template<typename T> T constant(T x) { return 1; }
	template<typename T> T dtanh(T x) { return 1 / cosh(x) / cosh(x); }
	template<typename T> T sign1(T x) { return x > 0 ? 1 : -1; }
	template<typename T> T dsign1(T x) { return 1; }
	template<typename T> T is(T x) { return x > 0.5 ? 1 : 0; }
	template<typename T> T dis(T x) { return 1; }
	template<typename T> T softplus(T x) { return log(1 + exp(x)); }
	template<typename T> T relu(T x) { return x > 0 ? x : 0; }
	template<typename T> T drelu(T x) { return x > 0 ? 1 : 0; }

	MYMATH_VECTOR(log_v, log);
	MYMATH_VECTOR(sigmoid_v, sigmoid);
	//MYMATH_VECTOR_B(sigmoid_vb, dsigmoid);
	//sigmoid导数直接使用y计算
	template<typename T> int sigmoid_vb(const T* a, const T* da, const T* x, T* dx, int size)
	{
		for (int i = 0; i < size; i++)
		{
			dx[i] = a[i] * (1 - a[i]) * da[i];
		}
		return 0;
	}

	MYMATH_VECTOR(tanh_v, tanh);
	template<typename T> int tanh_vb(const T* a, const T* da, const T* x, T* dx, int size)
	{
		for (int i = 0; i < size; i++)
		{
			dx[i] = (1 - a[i] * a[i]) * da[i];
		}
		return 0;
	}

	MYMATH_VECTOR(exp_v, exp);
	//MYMATH_VECTOR_B(exp_vb, exp);

	MYMATH_VECTOR(softplus_v, softplus);
	MYMATH_VECTOR_B(softplus_vb, sigmoid);

	MYMATH_VECTOR(relu_v, relu);
	MYMATH_VECTOR_B(relu_vb, drelu);

	template<typename T> int linear_v(T* x, T* a, int size) { memcpy(a, x, sizeof(T)*size); }
	MYMATH_VECTOR_B(linear_vb, constant);



	template<typename T> int nullfunction(T* x, T* a, int size) { return 0; }

	template<typename T> T conv(T* x, int x_stride, T* k, int k_stride, int w, int h)
	{
		T v = 0;
		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < h; j++)
			{
				v += x[i + j*x_stride] * k[i + j*k_stride];
			}
		}
		return v;
	}

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
	// void swap(int &a, int &b) { auto t = a; a = b; b = t; }
#undef MYMATH_FOR
#undef MYMATH_VECTOR
#undef MYMATH_FOR_B
#undef MYMATH_VECTOR_B

};

