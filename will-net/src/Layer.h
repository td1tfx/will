#pragma once
#include <vector>
#include <functional>
#include <string>
#include "Matrix.h"
#include "Option.h"

//���أ����룬���
typedef enum
{
	lt_Hidden,
	lt_Input,
	lt_Output,
} NeuralLayerType;

//��������
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

//�񾭲�
class Layer
{
public:
	Layer();
	virtual ~Layer();

	int Id;

	int OutputCountPerGroup;  //����ȫ���Ӳ㣬��������ڽڵ���������������ʽ���岻ͬ

	static int GroupCount;    //�������в���������һ��
	static void setGroupCount(int gc) { GroupCount = gc; }

	static int Step;  //��������

	NeuralLayerType Type = lt_Hidden;
	NeuralLayerConnectionType ConnetionType = lc_Full;

	bool NeedTrain = true;   //�������Ҫѵ����ôҲ���跴�򴫲�����ѵ����ʱ��Ҳֻ�輤��һ��
	void setNeedTrain(bool nt) { NeedTrain = nt; }

	//����ȫ���Ӿ����⼸��������ʽ��ͬ�������ǽڵ�������������������	
	
	Matrix* XMatrix = nullptr; //XMatrix�ռ���һ�����������������֮����Ǳ������A
	Matrix* dXMatrix = nullptr;
	Matrix* AMatrix = nullptr; //�������Ҫֱ������A
	Matrix* dAMatrix = nullptr;

	Matrix* X2Matrix = nullptr;
	Matrix* dX2Matrix = nullptr;
	
	Matrix* YMatrix = nullptr; //Y�൱�ڱ�׼�𰸣��������ʹ��

	int ImageRow = 1, ImageCol = 1, ImageCountPerGroup;

	//ֻ��������б�Ҫ������������������������õ���Ӧ��ֵ
	void setImageMode(int w, int h, int count);

	Layer *PrevLayer, *NextLayer;

	void deleteData();

	//active������ʽ
	ActiveFunctionType _activeFunctionType = af_Sigmoid;
	void setActiveFunction(ActiveFunctionType af) { _activeFunctionType = af; }

	CostFunctionType _costFunctionType = cf_CrossEntropy;
	void setCostFunction(CostFunctionType cf) { _costFunctionType = cf; }

	//���º���������ʹ�������������㣬���ز㲻����ʹ�ã�
	Matrix* getAMatrix() { return AMatrix; }
	Matrix* getYMatrix() { return YMatrix; }
	Matrix* getdAMatrix() { return dAMatrix; }
	real& getAValue(int x, int y) { return AMatrix->getData(x, y); }

	virtual void setSubType(ResampleType re) {}
	virtual void setSubType(ConvolutionType cv) {}

	//���淲�������������ģ����޺�׺�������й������֣��ڴ���׺�������Ǹ�������Ĺ���
	void resetGroupCount();
	void connetPrevlayer(Layer* prevLayer);
	void initData(NeuralLayerType type, NeuralLayerInitInfo* info) { this->Type = type; initData2(info); }
	void activeBackward();  //����ʵ��ֻ��������Ϊ������ʵ�֣������ۺ�������ʽ�������㽻�����Ե�����

	//�����ʵ����ֻ���������֣��������κ��㷨����ʹ�㷨���ظ��Ĳ�����Ȼ�����ദ����
	//�㷨�����updateDelta2��activeOutputValue��spreadDeltaToPrevLayer��backPropagate
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



