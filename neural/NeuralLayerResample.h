#pragma once
#include "NeuralLayer.h"
class NeuralLayerResample :
	public NeuralLayer
{
public:

	//�ڴ���ͼ��ģʽ��ʱ����һ��output�����������ﱻתΪ����
	int region_m = 2, region_n = 2;

	NeuralLayerResample();
	virtual ~NeuralLayerResample();

	int* maxPos = nullptr;   //��¼���ֵ��λ�ã����Ժ󿴿��ܲ��ܸĳ���cuda

	ResampleType _resampleType = re_Findmax;

protected:
	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void updateDelta2() override;
public:
	void activeOutputValue() override;
	void spreadDeltaToPrevLayer() override;
	void backPropagate(double learnSpeed, double lambda) override;
	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;

	void setSubType(ResampleType re) override { _resampleType = re; }
};

