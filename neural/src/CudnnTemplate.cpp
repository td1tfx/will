#include "CudnnTemplate.h"

#define CUDNN_CERATE_DESCRIPTOR(type) template<> cudnnStatus_t cudnnCreateDescriptor<cudnn##type##Descriptor_t>(cudnn##type##Descriptor_t* t)\
{return cudnnCreate##type##Descriptor(t);}
#define CUDNN_DESTROY_DESCRIPTOR(type) template<> cudnnStatus_t cudnnDestroyDescriptor<cudnn##type##Descriptor_t>(cudnn##type##Descriptor_t t)\
{auto r = CUDNN_STATUS_EXECUTION_FAILED; if (t) {r=cudnnDestroy##type##Descriptor(t); t=nullptr;} return r;}

#define CUDNN_DESCRIPTOR(type) CUDNN_CERATE_DESCRIPTOR(type)\
CUDNN_DESTROY_DESCRIPTOR(type)

CUDNN_DESCRIPTOR(Tensor)
CUDNN_DESCRIPTOR(Activation)
CUDNN_DESCRIPTOR(OpTensor)
CUDNN_DESCRIPTOR(Pooling)
CUDNN_DESCRIPTOR(Convolution)
CUDNN_DESCRIPTOR(Filter)
CUDNN_DESCRIPTOR(RNN)
CUDNN_DESCRIPTOR(Dropout)
CUDNN_DESCRIPTOR(SpatialTransformer)
CUDNN_DESCRIPTOR(LRN)

#undef CUDNN_CERATE_DESCRIPTOR
#undef CUDNN_DESTROY_DESCRIPTOR
#undef CUDNN_DESCRIPTOR


