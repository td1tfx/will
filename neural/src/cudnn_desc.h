#pragma once
#include "cudnn.h"
#include "cblas.h"

template <typename T>
inline cudnnStatus_t cudnnCreateDescriptor(T* desc);

template <typename T>
inline cudnnStatus_t cudnnDestroyDescriptor(T desc);

//#define CUDNN_CREATE_SET_DESCIPTOR(t, func) do {if(!t){cudnnCreateDescriptor(&t);(func);}} while(0)


