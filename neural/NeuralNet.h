#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include <cmath>
#include "lib/libconvert.h"
#include "Option.h"
#include "NeuralLayerFactory.h"
#include "Test.h"


//ѧϰģʽ
typedef enum
{
	nl_Whole = 0,          //ȫ������һ��ѧϰ�����ݶ��ʱ�����������
	nl_Online = 1,         //ÿ��ѧϰһ������ʵ����������Ĵ���
	nl_MiniBatch = 2,      //ÿ��ѧϰһС�����������һ����Ӱ����ܻ�Ƚϴ�
} NeuralNetLearnType;


//����ģʽ
typedef enum
{
	nw_Fit = 0,            //���
	nw_Classify = 1,       //���࣬��ɸѡ���ֵ��Ϊ1��������Ϊ0
	nw_Probability = 2,    //���ʣ�������һ��	
} NeuralNetWorkType;


//����
class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();
	int Id;

	int MaxGroup = 100000;  //һ���ܴ���������������ڴ���Դ��С���

	int InputNodeCount;
	int OutputNodeCount;

	//ѵ����
	Matrix* train_input = nullptr;
	Matrix* train_expect = nullptr;
	int train_groupCount = 0;
	//���Լ�
	Matrix* test_input = nullptr;
	Matrix* test_expect = nullptr;
	int test_groupCount = 0;

	void run(Option* option);

	//�񾭲�
	NeuralLayer** Layers;
	int LayerCount = 0;
	//std::vector<NeuralLayer*>& getLayerVector() { return Layers; }
	NeuralLayer*& getLayer(int number) { return Layers[number]; }
	NeuralLayer*& getFirstLayer() { return Layers[0]; }
	NeuralLayer*& getLastLayer() { return Layers[LayerCount - 1]; }
	int getLayerCount() { return LayerCount; };

	NeuralNetLearnType BatchMode = nl_Whole;
	int MiniBatchCount = -1;
	void setLearnType(NeuralNetLearnType lm, int lb = -1);

	real LearnSpeed = 0.5;  //ѧϰ�ٶ�
	void setLearnSpeed(real s) { LearnSpeed = s; }

	real Lambda = 0.0;      //���򻯲�������ֹ�����
	void setRegular(real l) { Lambda = l; }

	NeuralNetWorkType WorkType = nw_Fit;
	void setWorkType(NeuralNetWorkType wm);

	void createLayers(int layerCount);  //��������������

	void train(int times = 1000000, int interval = 1000, real tol = 1e-3, real dtol = 0);  //ѵ������

	void active(Matrix* input, Matrix* expect, Matrix* output, int groupCount, int batchCount,
		bool learn = false, real* error = nullptr);  //����һ�����

	//void setInputData(d_matrix* input, int groupid);
	//void setExpectData(d_matrix* expect, int groupid);

	void getOutputData(Matrix* output, int groupCount, int col = 0);

	void readData(const char* filename, int* count, Matrix** input, Matrix** expect);
	int resetGroupCount(int n);

	//��������
	virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //������������д����
	void saveInfo(const char* filename = nullptr);
	void createByLoad(const char* filename);

	//NeuralNetCalMode activeMode = ByNode;
	//NeuralNetCalMode backPropageteMode = ByNode;

	void readMNIST();

	void selectTest();
	void test(int forceOutput = 0, int testMax = 0);
	void extraTest(const char* filename, int forceOutput = 0, int testMax = 0);
	void outputTest(const char* info, int nodeCount, int groupCount, Matrix* input, Matrix* expect, int forceOutput, int testMax);
};

