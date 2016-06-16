#pragma once
#include <math.h>
#include <string.h>

#define MyMathFor(f) do{for(int i=0;i<size;i++){y[i]=f(x[i]);}return 0;}while(0)
#define dexp_v exp_v 
namespace MyMath
{
	static int min(int a, int b) { return a > b ? b : a; }
	static int max(int a, int b) { return a > b ? a : b; }

	static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
	static double dsigmoid(double x) { double a = 1 + exp(-x); return (a - 1) / (a*a); }
	static double constant(double x) { return 1; }
	static double dtanh(double x) { return 1 / cosh(x) / cosh(x); }
	static double sign1(double x) { return x > 0 ? 1 : -1; }
	static double dsign1(double x) { return 1; }
	static double is(double x) { return x > 0.5 ? 1 : 0; }
	static double dis(double x) { return 1; }

	static int sigmoid_v(double* x, double* y, int size) { MyMathFor(sigmoid); }
	static int dsigmoid_v(double* x, double* y, int size) { MyMathFor(dsigmoid); }
	static int linear_v(double* x, double* y, int size) { memcpy(y, x, sizeof(double)*size); }
	static int dlinear_v(double* x, double* y, int size) { MyMathFor(constant); }
	static int exp_v(double* x, double* y, int size) { MyMathFor(exp); }
	static int tanh_v(double* x, double* y, int size) { MyMathFor(tanh); }
	static int dtanh_v(double* x, double* y, int size) { MyMathFor(dtanh); }

	//static int min(int a, int b) { return (a < b) ? a : b; }
	//static void swap(int &a, int &b) { auto t = a; a = b; b = t; }

};

