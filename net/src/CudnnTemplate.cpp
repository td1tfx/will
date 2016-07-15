#include "CudnnTemplate.h"

#define CUDNN_CERATE_DESCRIPTOR(type) template<> \
cudnnStatus_t cudnnCreateDescriptor<cudnn##type##Descriptor_t>(cudnn##type##Descriptor_t* t) \
{return cudnnCreate##type##Descriptor(t);}
#define CUDNN_DESTROY_DESCRIPTOR(type) template<> \
cudnnStatus_t cudnnDestroyDescriptor<cudnn##type##Descriptor_t>(cudnn##type##Descriptor_t t) \
{auto r = CUDNN_STATUS_EXECUTION_FAILED; if (t) {r=cudnnDestroy##type##Descriptor(t); t=nullptr;} return r;}

#define CUDNN_DESCRIPTOR_PAIR(type) CUDNN_CERATE_DESCRIPTOR(type) \
CUDNN_DESTROY_DESCRIPTOR(type)

CUDNN_DESCRIPTOR_PAIR(Tensor)
CUDNN_DESCRIPTOR_PAIR(Activation)
CUDNN_DESCRIPTOR_PAIR(OpTensor)
CUDNN_DESCRIPTOR_PAIR(Pooling)
CUDNN_DESCRIPTOR_PAIR(Convolution)
CUDNN_DESCRIPTOR_PAIR(Filter)
CUDNN_DESCRIPTOR_PAIR(RNN)
CUDNN_DESCRIPTOR_PAIR(Dropout)
CUDNN_DESCRIPTOR_PAIR(SpatialTransformer)
CUDNN_DESCRIPTOR_PAIR(LRN)

#undef CUDNN_CERATE_DESCRIPTOR
#undef CUDNN_DESTROY_DESCRIPTOR
#undef CUDNN_DESCRIPTOR_PAIR


