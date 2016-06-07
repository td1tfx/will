#pragma once
#include <vector>
#include <functional>
#include <string>
#include "MyMath.h"
#include "MatrixFunctions.h"


//隐藏，输入，输出
typedef enum
{
	Hidden,
	Input,
	Output,
} NeuralLayerType;

typedef enum
{
	FullConnection,
	Convolution,
	Resample,
} NeuralLayerConnectionMode;

//神经层
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int Id;

	int OutputCount;  //对于全连接层，输出数等于节点数，对于其他形式定义不同
	static int GroupCount;   //对于所有层数据量都一样
	static int EffectiveGroupCount;  //必须小于数据组数，表示其后的是废数据
	static int Step;  //仅调试用

	NeuralLayerType Type = Hidden;
	NeuralLayerConnectionMode WorkMode = FullConnection;

	bool NeedTrain = true;   //如果不需要训练那么也无需反向传播，在训练的时候也只需激活一次
	void setNeedTrain(bool nt) { NeedTrain = nt; }

	//对于全连接矩阵，这几个矩阵形式相同，行数是节点数，列数是数据组数
	//Expect仅输出层使用，输入层需要直接设置Output
	d_matrix *InputMatrix = nullptr, *OutputMatrix = nullptr, *DeltaMatrix = nullptr, *ExpectMatrix = nullptr;
	//weight矩阵，对于全连接层，行数是本层的节点数，列数是上一层的节点数
	d_matrix* WeightMatrix = nullptr;
	//偏移向量，维度为本层节点数
	d_matrix* BiasVector = nullptr;
	//更新偏移向量的辅助向量，所有值为1，维度为数据组数
	d_matrix* _asBiasVector = nullptr;

	NeuralLayer *PrevLayer, *NextLayer;

	void deleteData();

	//dactive是active的导数
	ActiveFunctionMode _activeMode = Sigmoid;
	void setActiveFunction(ActiveFunctionMode afm) { _activeMode = afm; }

	virtual void initData(int nodeCount, int groupCount, NeuralLayerType type = Hidden) {}
	virtual void resetData(int groupCount) {}
	virtual void connetPrevlayer(NeuralLayer* prevLayer) {}
	virtual void activeOutputValue() {}
	virtual void updateDelta() {}
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int readInfo(double* v, int n) { return 0; }

	//以下函数仅建议使用在输入和输出层，隐藏层不建议使用！
	d_matrix* getOutputMatrix() { return OutputMatrix; }
	d_matrix* getExpectMatrix() { return ExpectMatrix; }
	d_matrix* getDeltaMatrix() { return DeltaMatrix; }
	double& getOutputValue(int x, int y) { return OutputMatrix->getData(x, y); }

};



