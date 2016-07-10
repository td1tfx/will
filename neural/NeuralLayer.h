#pragma once
#include <vector>
#include <functional>
#include <string>
#include "MyMath.h"
#include "Matrix.h"


//隐藏，输入，输出
typedef enum
{
	lt_Hidden,
	lt_Input,
	lt_Output,
} NeuralLayerType;

//连接类型
typedef enum
{
	lc_Full,
	lc_Convolution,
	lc_Resample,
} NeuralLayerConnectionType;

//神经层
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int Id;

	int OutputCountPerGroup;  //对于全连接层，输出数等于节点数，对于其他形式定义不同

	static int GroupCount;    //对于所有层数据量都一样
	static void setGroupCount(int gc) { GroupCount = gc; }

	static int Step;  //仅调试用

	NeuralLayerType Type = lt_Hidden;
	NeuralLayerConnectionType ConnetionType = lc_Full;

	bool NeedTrain = true;   //如果不需要训练那么也无需反向传播，在训练的时候也只需激活一次
	void setNeedTrain(bool nt) { NeedTrain = nt; }

	//对于全连接矩阵，这几个矩阵形式相同，行数是节点数，列数是数据组数
	//Expect仅输出层使用，输入层需要直接设置Y
	//XMatrix收集上一层的输出，激活函数作用之后就是本层输出
	Matrix *XMatrix = nullptr, *AMatrix = nullptr;
	Matrix *dXMatrix = nullptr, *dAMatrix = nullptr;
	Matrix* YMatrix = nullptr;

	int ImageRow = 1, ImageCol = 1, ImageCountPerGroup;

	//只有输入层有必要调用这个函数，其他层均计算得到对应的值
	void setImageMode(int w, int h, int count);

	NeuralLayer *PrevLayer, *NextLayer;

	void deleteData();

	//active函数形式
	ActiveFunctionType _activeFunctionType = af_Sigmoid;
	void setActiveFunction(ActiveFunctionType af) { _activeFunctionType = af; }

	CostFunctionType _costFunctionType = cf_CrossEntropy;
	void setCostFunction(CostFunctionType cf) { _costFunctionType = cf; }

	//以下函数仅建议使用在输入和输出层，隐藏层不建议使用！
	Matrix* getOutputMatrix() { return AMatrix; }
	Matrix* getExpectMatrix() { return YMatrix; }
	Matrix* getDeltaMatrix() { return dAMatrix; }
	real& getOutputValue(int x, int y) { return AMatrix->getData(x, y); }

	virtual void setSubType(ResampleType re) {}
	virtual void setSubType(ConvolutionType cv) {}

	//下面凡是有两个函数的，在无后缀函数中有公共部分，在带后缀函数中是各自子类的功能
	void resetGroupCount();
	void connetPrevlayer(NeuralLayer* prevLayer);
	void initData(NeuralLayerType type, int x1, int x2) { this->Type = type; initData2(x1, x2); }
	void updateDelta();  //这里实际只包含了作为输出层的实现，即代价函数的形式，其他层交给各自的子类

	//基类的实现里只处理公共部分，不处理任何算法，即使算法有重复的部分仍然在子类处理！！
	//算法相关是updateDelta2，activeOutputValue，spreadDeltaToPrevLayer，backPropagate
protected:
	virtual void initData2(int x1, int x2) {}
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2() {}
	virtual void updateDelta2() {}
public:
	virtual void activeOutput() {}
	virtual void spreadDeltaToPrevLayer() {}
	virtual void updateWeightBias(real learnSpeed, real lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(real* v, int n) { return 0; }

};



