#pragma once
#include "NeuralLayer.h"
class NeuralLayerFull :
	public NeuralLayer
{
public:
	NeuralLayerFull();
	virtual ~NeuralLayerFull();

	//weight矩阵，对于全连接层，行数是本层的节点数，列数是上一层的节点数
	d_matrix* WeightMatrix = nullptr;
	//偏移向量，维度为本层节点数
	d_matrix* BiasVector = nullptr;
	//更新偏移向量的辅助向量，所有值为1，维度为数据组数
	d_matrix* _asBiasVector = nullptr;

	void initData2(int x1, int x2) override;
	void resetGroupCount2() override;
	void connetPrevlayer2() override;
	void activeOutputValue() override;
	void updateDelta2() override;
	void spreadDeltaToPrevLayer() override;
	void backPropagate(double learnSpeed, double lambda) override;

	int saveInfo(FILE* fout) override;
	int loadInfo(double* v, int n) override;
};

