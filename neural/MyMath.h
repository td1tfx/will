#pragma once
#include <math.h>
#include <string.h>

#define MyMathFor(f) do{for(int i=0;i<size;i++){y[i]=f(x[i]);}}while(0)
#define dexp_v exp_v 
namespace MyMath
{
	static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
	static double dsigmoid(double x) { double a = 1 + exp(-x); return (a - 1) / (a*a); }
	static double constant(double x) { return 1; }
	static double dtanh(double x) { return 1 / cosh(x) / cosh(x); }
	static double sign1(double x) { return x > 0 ? 1 : -1; }
	static double dsign1(double x) { return 1; }
	static double is(double x) { return x > 0.5 ? 1 : 0; }
	static double dis(double x) { return 1; }

	static void sigmoid_v(double* x, double* y, int size) { MyMathFor(sigmoid); }
	static void dsigmoid_v(double* x, double* y, int size) { MyMathFor(dsigmoid); }
	static void linear_v(double* x, double* y, int size) { memcpy(y, x, sizeof(double)*size); }
	static void dlinear_v(double* x, double* y, int size) { MyMathFor(constant); }
	static void exp_v(double* x, double* y, int size) { MyMathFor(exp); }
	static void tanh_v(double* x, double* y, int size) { MyMathFor(tanh); }
	static void dtanh_v(double* x, double* y, int size) { MyMathFor(dtanh); }

	//static void swap(int &a, int &b) { auto t = a; a = b; b = t; }
};

