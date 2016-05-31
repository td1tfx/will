#pragma once
extern "C"
{
#include "cblas.h"
}
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

struct d_matrix
{
private:
	double* data = nullptr;
	int m=0;
	int n=0;
	int max_script;
	bool insideData = true;
public:
	d_matrix(int x, int y, bool insideData = true)
	{
		m = x;
		n = y;
		this->insideData = insideData;
		if (insideData)
			data = new double[m*n + 1];
		max_script = m*n;
	}
	~d_matrix()
	{
		if(insideData) delete[] data;
	}
	int getRow()
	{
		return m;
	}
	int getCol()
	{
		return n;
	}
	double& getData(int x, int y)
	{
		return data[std::min(x + y*m, max_script)];
	}
	double& getData(int i)
	{
		return data[std::min(i, max_script)];
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
	//����������ܲ���ȫ�����ã���
	void resetDataPointer(double* d)
	{
		data = d;
	}
	double& operator [] (int i)
	{
		return data[i];
	}
	double ddot();

	void print();
	void memcpyDataIn(double* src, int size);
	void memcpyDataOut(double* dst, int size);
	void expand();
	int indexRowMaxAbs(int r);

	void initData(double v);
	void initRandom();
	void multiply(double v);

	static void cpyData(d_matrix* dst, d_matrix* src);

	static void product(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, CBLAS_TRANSPOSE ta = CblasNoTrans, CBLAS_TRANSPOSE tb = CblasNoTrans);
	static void productVector(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, CBLAS_TRANSPOSE ta = CblasNoTrans);
	static void hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R);
	static void minus(d_matrix* A, d_matrix* B, d_matrix* R);
};


