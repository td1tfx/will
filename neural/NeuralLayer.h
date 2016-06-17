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
	static void setGroupCount(int gc) { GroupCount = gc; } 

	static int Step;  //��������

	NeuralLayerType Type = Hidden;
	NeuralLayerConnectionMode WorkMode = FullConnection;

	bool NeedTrain = true;   //�������Ҫѵ����ôҲ���跴�򴫲�����ѵ����ʱ��Ҳֻ�輤��һ��
	void setNeedTrain(bool nt) { NeedTrain = nt; }

	//����ȫ���Ӿ����⼸��������ʽ��ͬ�������ǽڵ�������������������
	//Expect�������ʹ�ã��������Ҫֱ������Output
	//UnactivedMatrix�ռ���һ�����������������֮����Ǳ������
	d_matrix *UnactivedMatrix = nullptr, *OutputMatrix = nullptr, *DeltaMatrix = nullptr, *ExpectMatrix = nullptr;

	int ImageRow=1, ImageCol=1, ImageCount;

	//ֻ��������б�Ҫ������������������������õ���Ӧ��ֵ
	void setImageMode(int w, int h, int count);

	NeuralLayer *PrevLayer, *NextLayer;

	void deleteData();

	//dactive��active�ĵ���
	ActiveFunctionMode ActiveMode = Sigmoid;
	void setActiveFunction(ActiveFunctionMode afm) { ActiveMode = afm; }
	void connetPrevlayer(NeuralLayer* prevLayer);
	void resetGroupCount();
	void initData(NeuralLayerType type, int x1, int x2) { this->Type = type; initData2(x1, x2); }

protected:
	virtual void resetGroupCount2() {}
	virtual void connetPrevlayer2() {}
	virtual void initData2(int x1, int x2) {}
public:	
	virtual void activeOutputValue() {}
	virtual void updateDelta() {}
	virtual void spreadDeltaToPrevLayer() {}
	virtual void backPropagate(double learnSpeed, double lambda) {}
	virtual int saveInfo(FILE* fout) { return 0; }
	virtual int loadInfo(double* v, int n) { return 0; }

	//���º���������ʹ�������������㣬���ز㲻����ʹ�ã�
	d_matrix* getOutputMatrix() { return OutputMatrix; }
	d_matrix* getExpectMatrix() { return ExpectMatrix; }
	d_matrix* getDeltaMatrix() { return DeltaMatrix; }
	double& getOutputValue(int x, int y) { return OutputMatrix->getData(x, y); }

};



