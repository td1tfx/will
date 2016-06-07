#pragma once

#ifdef __cplusplus   
#define HBAPI extern "C" __declspec (dllimport)   
#else   
#define HBAPI __declspec (dllimport)   
#endif   

int _stdcall cuda_hadamardProduct(const double *A, const double *B, double *R, unsigned int size);
int _stdcall cuda_sigmoid(double *A, double *B, unsigned int size);
int _stdcall cuda_dsigmoid(double *A, double *B, unsigned int size);
int _stdcall cuda_exp(double *A, double *B, unsigned int size);
