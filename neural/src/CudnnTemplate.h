#pragma once
#include "cudnn.h"

template <typename T> cudnnStatus_t cudnnCreateDescriptor(T* desc);
template <typename T> cudnnStatus_t cudnnDestroyDescriptor(T desc);

#define CUDNN_CREATE_SET_DESCIPTOR(t, func) do {if(!t){cudnnCreateDescriptor(&t);(func);}} while(0)