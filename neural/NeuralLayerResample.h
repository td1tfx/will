#pragma once
#include "NeuralLayer.h"
class NeuralLayerResample :
	public NeuralLayer
{
public:

	//�ڴ���ͼ��ģʽ��ʱ����һ��output�����������ﱻתΪ����
	d_matrix* outputMatrix_image = nullptr;

	NeuralLayerResample();
	virtual ~NeuralLayerResample();
};

