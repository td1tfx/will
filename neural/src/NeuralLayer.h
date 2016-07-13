#pragma once
#include <vector>
#include <functional>
#include <string>
#include "Matrix.h"

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
} NeuralLayerConnectionType;

typedef union
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
} NeuralLayerInitInfo;

//�񾭲�
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

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
	//Expect�������ʹ�ã��������Ҫֱ������Y
	//XMatrix�ռ���һ�����������������֮����Ǳ������
	Matrix *XMatrix = nullptr, *AMatrix = nullptr;
	Matrix *dXMatrix = nullptr, *dAMatrix = nullptr;
	Matrix* YMatrix = nullptr;

	int ImageRow = 1, ImageCol = 1, ImageCountPerGroup;

	//ֻ��������б�Ҫ������������������������õ���Ӧ��ֵ
	void setImageMode(int w, int h, int count);

	NeuralLayer *PrevLayer, *NextLayer;

	void deleteData();

	//active������ʽ
	ActiveFunctionType _activeFunctionType = af_Sigmoid;
	void setActiveFunction(ActiveFunctionType af) { _activeFunctionType = af; }

	CostFunctionType _costFunctionType = cf_CrossEntropy;
	void setCostFunction(CostFunctionType cf) { _costFunctionType = cf; }

	//���º���������ʹ�������������㣬���ز㲻����ʹ�ã�
	Matrix* getOutputMatrix() { return AMatrix; }
	Matrix* getExpectMatrix() { return YMatrix; }
	Matrix* getDeltaMatrix() { return dAMatrix; }
	real& getOutputValue(int x, int y) { return AMatrix->getData(x, y); }

	virtual void setSubType(ResampleType re) {}
	virtual void setSubType(ConvolutionType cv) {}

	//���淲�������������ģ����޺�׺�������й������֣��ڴ���׺�������Ǹ�������Ĺ���
	void resetGroupCount();
	void connetPrevlayer(NeuralLayer* prevLayer);
	void initData(NeuralLayerType type, NeuralLayerInitInfo* info) { this->Type = type; initData2(info); }
	void updateDelta();  //����ʵ��ֻ��������Ϊ������ʵ�֣������ۺ�������ʽ�������㽻�����Ե�����

	//�����ʵ����ֻ���������֣��������κ��㷨����ʹ�㷨���ظ��Ĳ�����Ȼ�����ദ����
	//�㷨�����updateDelta2��activeOutputValue��spreadDeltaToPrevLayer��backPropagate
protected:
	virtual void initData2(NeuralLayerInitInfo* info) {}
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



