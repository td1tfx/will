#pragma once
#include <vector>
#include <functional>
#include <string>
#include "Matrix.h"
#include "Option.h"

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
	lc_Pooling,
	lc_BatchNormalization,
} NeuralLayerConnectionType;

struct NeuralLayerInitInfo
{
	struct
	{
		int outputCount;
	} full;
	struct { } Convolution;
	struct
	{
		int window_w, window_h;
		int stride_w, stride_h;
	} pooling;
	int initWithOption(Option* op);
};

//神经层
class Layer
{
public:
	Layer();
	virtual ~Layer();

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
	
	Matrix* XMatrix = nullptr; //XMatrix收集上一层的输出，激活函数作用之后就是本层输出A
	Matrix* dXMatrix = nullptr;
	Matrix* AMatrix = nullptr; //输入层需要直接设置A
	Matrix* dAMatrix = nullptr;

	Matrix* X2Matrix = nullptr;
	Matrix* dX2Matrix = nullptr;
	
	Matrix* YMatrix = nullptr; //Y相当于标准答案，仅输出层使用

	int ImageRow = 1, ImageCol = 1, ImageCountPerGroup;

	//只有输入层有必要调用这个函数，其他层均计算得到对应的值
	void setImageMode(int w, int h, int count);

	Layer *PrevLayer, *NextLayer;

	void deleteData();

	//active函数形式
	ActiveFunctionType _activeFunctionType = af_Sigmoid;
	void setActiveFunction(ActiveFunctionType af) { _activeFunctionType = af; }

	CostFunctionType _costFunctionType = cf_CrossEntropy;
	void setCostFunction(CostFunctionType cf) { _costFunctionType = cf; }

	//以下函数仅建议使用在输入和输出层，隐藏层不建议使用！
	Matrix* getAMatrix() { return AMatrix; }
	Matrix* getYMatrix() { return YMatrix; }
	Matrix* getdAMatrix() { return dAMatrix; }
	real& getAValue(int x, int y) { return AMatrix->getData(x, y); }

	virtual void setSubType(ResampleType re) {}
	virtual void setSubType(ConvolutionType cv) {}

	//下面凡是有两个函数的，在无后缀函数中有公共部分，在带后缀函数中是各自子类的功能
	void resetGroupCount();
	void connetPrevlayer(Layer* prevLayer);
	void initData(NeuralLayerType type, NeuralLayerInitInfo* info) { this->Type = type; initData2(info); }
	void activeBackward();  //这里实际只包含了作为输出层的实现，即代价函数的形式，其他层交给各自的子类

	//基类的实现里只处理公共部分，不处理任何算法，即使算法有重复的部分仍然在子类处理！！
	//算法相关是updateDelta2，activeOutputValue，spreadDeltaToPrevLayer，backPropagate
protected:
	virtual void initData2(NeuralLayerInitInfo* info) {}
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2() {}
	virtual void activeBackward2() {}
public:
	virtual void activeForward() {}
	virtual void spreadDeltaToPrevLayer() {}
	virtual void updateParameters(real learnSpeed, real lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(real* v, int n) { return 0; }

};



