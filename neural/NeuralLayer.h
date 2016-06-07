#pragma once
#include <vector>
#include <functional>
#include <string>
#include "MyMath.h"
#include "MatrixFunctions.h"


//���أ����룬���
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

//�񾭲�
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int Id;

	int OutputCount;  //����ȫ���Ӳ㣬��������ڽڵ���������������ʽ���岻ͬ
	static int GroupCount;   //�������в���������һ��
	static int EffectiveGroupCount;  //����С��������������ʾ�����Ƿ�����
	static int Step;  //��������

	NeuralLayerType Type = Hidden;
	NeuralLayerConnectionMode WorkMode = FullConnection;

	bool NeedTrain = true;   //�������Ҫѵ����ôҲ���跴�򴫲�����ѵ����ʱ��Ҳֻ�輤��һ��
	void setNeedTrain(bool nt) { NeedTrain = nt; }

	//����ȫ���Ӿ����⼸��������ʽ��ͬ�������ǽڵ�������������������
	//Expect�������ʹ�ã��������Ҫֱ������Output
	d_matrix *InputMatrix = nullptr, *OutputMatrix = nullptr, *DeltaMatrix = nullptr, *ExpectMatrix = nullptr;
	//weight���󣬶���ȫ���Ӳ㣬�����Ǳ���Ľڵ�������������һ��Ľڵ���
	d_matrix* WeightMatrix = nullptr;
	//ƫ��������ά��Ϊ����ڵ���
	d_matrix* BiasVector = nullptr;
	//����ƫ�������ĸ�������������ֵΪ1��ά��Ϊ��������
	d_matrix* _asBiasVector = nullptr;

	NeuralLayer *PrevLayer, *NextLayer;

	void deleteData();

	//dactive��active�ĵ���
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

	//���º���������ʹ�������������㣬���ز㲻����ʹ�ã�
	d_matrix* getOutputMatrix() { return OutputMatrix; }
	d_matrix* getExpectMatrix() { return ExpectMatrix; }
	d_matrix* getDeltaMatrix() { return DeltaMatrix; }
	double& getOutputValue(int x, int y) { return OutputMatrix->getData(x, y); }

};



