#pragma once
#include <math.h>
#include <string.h>

namespace MyMath
{
#define MyMathFor(f) do{for(int i=0;i<size;i++){y[i]=f(x[i]);}}while(0)
#define MyMathVector(fv, f) static void fv(double* x, double* y, int size) { MyMathFor(f); }

	static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
	static double dsigmoid(double x) { double a = 1 + exp(-x); return (a - 1) / (a*a); }
	static double constant(double x) { return 1; }
	static double dtanh(double x) { return 1 / cosh(x) / cosh(x); }
	static double sign1(double x) { return x > 0 ? 1 : -1; }
	static double dsign1(double x) { return 1; }
	static double is(double x) { return x > 0.5 ? 1 : 0; }
	static double dis(double x) { return 1; }
	static double softplus(double x) { return log(1 + exp(x)); }
	static double relu(double x) { return x > 0 ? x : 0; }
	static double drelu(double x) { return x > 0 ? 1 : 0; }


	MyMathVector(sigmoid_v, sigmoid);
	MyMathVector(dsigmoid_v, dsigmoid);

	MyMathVector(tanh_v, tanh);
	MyMathVector(dtanh_v, dtanh);

	MyMathVector(exp_v, exp);
	MyMathVector(dexp_v, exp);

	MyMathVector(softplus_v, softplus);
	MyMathVector(dsoftplus_v, sigmoid);

	MyMathVector(relu_v, relu);
	MyMathVector(drelu_v, drelu);

	static void linear_v(double* x, double* y, int size) { memcpy(y, x, sizeof(double)*size); }
	MyMathVector(dlinear_v, constant);

	//static void swap(int &a, int &b) { auto t = a; a = b; b = t; }
#undef MyMathFor
#undef MyMathVector
};

