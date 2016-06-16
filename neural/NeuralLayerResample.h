#pragma once
#include "NeuralLayer.h"
class NeuralLayerResample :
	public NeuralLayer
{
public:

	//在处理图像模式的时候，上一层output的向量在这里被转为矩阵
	d_matrix* outputMatrix_image = nullptr;

	NeuralLayerResample();
	virtual ~NeuralLayerResample();
};

