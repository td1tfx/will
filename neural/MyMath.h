#pragma once
#include <math.h>
#include <string.h>
#include "types.h"

namespace MyMath
{
#define MYMATH_FOR(f) do{for(int i=0;i<size;i++){y[i]=f(x[i]);}}while(0)
#define MYMATH_VECTOR(fv, f) static int fv(real* x, real* y, int size) { MYMATH_FOR(f); return 0; }
#define MYMATH_FOR_B(f) do{for(int i=0;i<size;i++){y[i]=f(x[i])*y[i];}}while(0)
#define MYMATH_VECTOR_B(fv, f) static int fv(real* x, real* y, int size) { MYMATH_FOR_B(f); return 0; }

	static real sigmoid(real x) { return 1 / (1 + exp(-x)); }
	static real dsigmoid(real x) { real a = 1 + exp(-x); return (a - 1) / (a*a); }
	static real dsigmoid2(real y) { return y*(1-y); }
	static real constant(real x) { return 1; }
	static real dtanh(real x) { return 1 / cosh(x) / cosh(x); }
	static real sign1(real x) { return x > 0 ? 1 : -1; }
	static real dsign1(real x) { return 1; }
	static real is(real x) { return x > 0.5 ? 1 : 0; }
	static real dis(real x) { return 1; }
	static real softplus(real x) { return log(1 + exp(x)); }
	static real relu(real x) { return x > 0 ? x : 0; }
	static real drelu(real x) { return x > 0 ? 1 : 0; }


	MYMATH_VECTOR(sigmoid_v, sigmoid);
	MYMATH_VECTOR_B(sigmoid_vb, dsigmoid);
	MYMATH_VECTOR_B(sigmoid_vb2, dsigmoid2);

	MYMATH_VECTOR(tanh_v, tanh);
	MYMATH_VECTOR_B(tanh_vb, dtanh);

	MYMATH_VECTOR(exp_v, exp);
	MYMATH_VECTOR_B(exp_vb, exp);

	MYMATH_VECTOR(softplus_v, softplus);
	MYMATH_VECTOR_B(softplus_vb, sigmoid);

	MYMATH_VECTOR(relu_v, relu);
	MYMATH_VECTOR_B(relu_vb, drelu);

	static int linear_v(real* x, real* y, int size) { memcpy(y, x, sizeof(real)*size); }
	MYMATH_VECTOR_B(linear_vb, constant);

	static int nullfunction(real* x, real* y, int size) { return 0; }

	//static void swap(int &a, int &b) { auto t = a; a = b; b = t; }
#undef MYMATH_FOR
#undef MYMATH_VECTOR
#undef MYMATH_FOR_B
#undef MYMATH_VECTOR_B

};

