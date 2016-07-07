#pragma once
#include <math.h>
#include <string.h>
#include "types.h"

namespace MyMath
{
#define MyMathFor(f) do{for(int i=0;i<size;i++){y[i]=f(x[i]);}}while(0)
#define MyMathVector(fv, f) static int fv(real* x, real* y, int size) { MyMathFor(f); return 0; }
#define MyMathFor_b(f) do{for(int i=0;i<size;i++){y[i]=f(x[i])*y[i];}}while(0)
#define MyMathVector_b(fv, f) static int fv(real* x, real* y, int size) { MyMathFor_b(f); return 0; }

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


	MyMathVector(sigmoid_v, sigmoid);
	MyMathVector_b(sigmoid_vb, dsigmoid);
	MyMathVector_b(sigmoid_vb2, dsigmoid2);

	MyMathVector(tanh_v, tanh);
	MyMathVector_b(tanh_vb, dtanh);

	MyMathVector(exp_v, exp);
	MyMathVector_b(exp_vb, exp);

	MyMathVector(softplus_v, softplus);
	MyMathVector_b(softplus_vb, sigmoid);

	MyMathVector(relu_v, relu);
	MyMathVector_b(relu_vb, drelu);

	static int linear_v(real* x, real* y, int size) { memcpy(y, x, sizeof(real)*size); }
	MyMathVector_b(linear_vb, constant);

	static int nullfunction(real* x, real* y, int size) { return 0; }

	//static void swap(int &a, int &b) { auto t = a; a = b; b = t; }
#undef MyMathFor
#undef MyMathVector
#undef MyMathFor_b
#undef MyMathVector_b

};

