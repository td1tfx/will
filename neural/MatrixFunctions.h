#pragma once
extern "C"
{
#include "cblas.h"
}
#include <stdio.h>

struct d_matrix
{
private:
	double* data = nullptr;
	int m;
	int n;
public:
	d_matrix(int x, int y)
	{
		m = x;
		n = y;
		data = new double[m*n+1];
	}
	~d_matrix()
	{
		delete[] data;
	}
	double& getData(int x, int y)
	{
		return data[x + y*m];
	}
	double& getData(int i)
	{
		return data[i];
	}
	double* getDataPointer(int x, int y)
	{
		return &getData(x, y);
	}
	double* getDataPointer(int i)
	{
		return &getData(i);
	}
	double* getDataPointer()
	{
		return data;
	}
	double& operator [] (int i)
	{
		return data[i];
	}

	void print();

	void memcpyData(double* src, int size);

	static void product(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, CBLAS_TRANSPOSE ta = CblasNoTrans, CBLAS_TRANSPOSE tb = CblasNoTrans);
	static void hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R);
	static void minus(d_matrix* A, d_matrix* B, d_matrix* R);
};


