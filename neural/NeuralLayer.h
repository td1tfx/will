#pragma once
#include <vector>
#include <functional>
#include <string>
#include "MyMath.h"
#include "MatrixFunctions.h"


//���أ����룬���
typedef enum
{
	lt_Hidden,
	lt_Input,
	lt_Output,
} NeuralLayerType;

typedef enum
{
	lc_Full,
	lc_Convolution,
	lc_Resample,
} NeuralLayerConnectionType;

//�񾭲�
//�����ʵ����ֻ���������֣��������κ��㷨����ʹ�㷨���ظ��Ĳ�����Ȼ�����ദ����
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
	//Expect�������ʹ�ã��������Ҫֱ������Output
	//UnactivedMatrix�ռ���һ�����������������֮����Ǳ������
	d_matrix *UnactivedMatrix = nullptr, *OutputMatrix = nullptr, *DeltaMatrix = nullptr, *ExpectMatrix = nullptr;

	int ImageRow = 1, ImageCol = 1, ImageCountPerGroup;

	//ֻ��������б�Ҫ������������������������õ���Ӧ��ֵ
	void setImageMode(int w, int h, int count);

	NeuralLayer *PrevLayer, *NextLayer;

	void deleteData();

	//dactive��active�ĵ���
	ActiveFunctionType _activeFunctionType = af_Sigmoid;
	void setActiveFunction(ActiveFunctionType afm) { _activeFunctionType = afm; }

	//���º���������ʹ�������������㣬���ز㲻����ʹ�ã�
	d_matrix* getOutputMatrix() { return OutputMatrix; }
	d_matrix* getExpectMatrix() { return ExpectMatrix; }
	d_matrix* getDeltaMatrix() { return DeltaMatrix; }
	double& getOutputValue(int x, int y) { return OutputMatrix->getData(x, y); }

	virtual void setSubType(ResampleType re) {}
	virtual void setSubType(ConvolutionType cv) {}

	//���淲�������������ģ����޺�׺�������й������֣��ڴ���׺�������Ǹ�������Ĺ���
	void resetGroupCount();
	void connetPrevlayer(NeuralLayer* prevLayer);
	void initData(NeuralLayerType type, int x1, int x2) { this->Type = type; initData2(x1, x2); }
	void updateDelta();  //����ʵ��ֻ��������Ϊ������ʵ�֣������ۺ�������ʽ�������㽻�����Ե�����

protected:
	virtual void initData2(int x1, int x2) {}
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2() {}
	virtual void updateDelta2() {}
public:
	virtual void activeOutputValue() {}
	virtual void spreadDeltaToPrevLayer() {}
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(double* v, int n) { return 0; }

};



