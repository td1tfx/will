#include "NeuralNet.h"



NeuralNet::NeuralNet()
{
}


NeuralNet::~NeuralNet()
{
	for (auto& layer : getLayerVector())
	{
		delete layer;
	}
	if (inputData) delete[] inputData;
	if (expectData)	delete[] expectData;
	if (inputTestData) delete[] inputTestData;
	if (expectTestData) delete[] expectTestData;
}


//设置学习模式
void NeuralNet::setLearnMode(NeuralNetLearnMode lm)
{
	learnMode = lm;
}

//创建神经层
void NeuralNet::createLayers(int amount)
{
	layers.resize(amount);
	for (int i = 0; i < amount; i++)
	{
		auto layer = new NeuralLayer();
		layer->id = i;
		layers[i] = layer;
	}
}


//这里拆一部分数据为测试数据，写法有hack性质
void NeuralNet::selectTest()
{
	//备份原来的数据
	auto input = new double[inputAmount*realDataAmount];
	auto output = new double[outputAmount*realDataAmount];
	memcpy(input, inputData, sizeof(double)*inputAmount*realDataAmount);
	memcpy(output, expectData, sizeof(double)*outputAmount*realDataAmount);

	inputTestData = new double[inputAmount*realDataAmount];
	expectTestData = new double[outputAmount*realDataAmount];

	isTest.resize(realDataAmount);
	testDataAmount = 0;
	int p = 0, p_data = 0, p_test = 0;
	int it = 0, id = 0;
	for (int i = 0; i < realDataAmount; i++)
	{
		isTest[i] = (0.9 < 1.0*rand() / RAND_MAX);
		if (isTest[i])
		{
			memcpy(inputTestData + inputAmount*it, input+inputAmount*i, sizeof(double)*inputAmount);
			memcpy(expectTestData + outputAmount*it, output + outputAmount*i, sizeof(double)*outputAmount);
			testDataAmount++;
			it++;
		}
		else
		{
			memcpy(inputData + inputAmount*id, input + inputAmount*i, sizeof(double)*inputAmount);
			memcpy(expectData + outputAmount*id, output + outputAmount*i, sizeof(double)*outputAmount);
			id++;
		}
	}
	realDataAmount -= testDataAmount;
}

//输出拟合的结果和测试集的结果
void NeuralNet::test()
{

	auto output_train = new double[outputAmount*realDataAmount];

	//输出全部数据
	activeOutputValue(inputData, output_train, realDataAmount);
	fprintf(stdout, "\n%d groups train data comparing with expection:\n---------------------------------------\n", realDataAmount);
	for (int i = 0; i < realDataAmount; i++)
	{
		for (int j = 0; j < outputAmount; j++)
		{
			fprintf(stdout, "%8.4lf -->%8.4lf\t", output_train[i*outputAmount + j], expectData[i*outputAmount + j]);
		}
		fprintf(stdout, "\n");
	}
	delete [] output_train;

	if (testDataAmount <= 0) return;
	auto output_test = new double[outputAmount*testDataAmount];
	activeOutputValue(inputTestData, output_test, testDataAmount);
	fprintf(stdout, "\n%d groups test data:\n---------------------------------------\n", testDataAmount);
	for (int i = 0; i < testDataAmount; i++)
	{
		for (int j = 0; j < outputAmount; j++)
		{
			fprintf(stdout, "%8.4lf -->%8.4lf\t", output_test[i*outputAmount + j], expectTestData[i*outputAmount + j]);
		}
		fprintf(stdout, "\n");
	}
	delete [] output_test;
}

//计算输出
//这里按照前面的设计应该是逐步回溯计算，使用栈保存计算的顺序，待完善后修改
void NeuralNet::activeOutputValue(double* input, double* output, int amount)
{	
	if (input)
	{
		setInputData(input, inputAmount, amount);
	}	
	for (int i = 1; i < getLayerAmount(); i++)
	{
		layers[i]->activeOutputValue();
	}
	if (output)
	{
		getOutputData(output, outputAmount, amount);
	}
}

void NeuralNet::setInputData(double* input, int nodeAmount, int groupAmount)
{
	for (int i_node = 0; i_node < nodeAmount; i_node++)
	{
		for (int i_group = 0; i_group < groupAmount; i_group++)
		{
			getFirstLayer()->getOutput(i_node, i_group) = input[i_node + i_group*nodeAmount];
		}
	}
}

void NeuralNet::getOutputData(double* output, int nodeAmount, int groupAmount)
{
	for (int i_node = 0; i_node < nodeAmount; i_node++)
	{
		for (int i_group = 0; i_group < groupAmount; i_group++)
		{
			output[i_node + i_group*nodeAmount] = getLastLayer()->getOutput(i_node, i_group);
		}
	}
}

void NeuralNet::setExpectData(double* expect, int nodeAmount, int groupAmount)
{
	for (int i_node = 0; i_node < nodeAmount; i_node++)
	{
		for (int i_group = 0; i_group < groupAmount; i_group++)
		{
			getLastLayer()->getExpect(i_node, i_group) = expect[i_node + i_group*nodeAmount];
		}
	}
}

//学习过程
void NeuralNet::learn()
{
	//正向
	for (int i = 1; i < getLayerAmount(); i++)
	{
		layers[i]->activeOutputValue();
	}
	//反向传播
	for (int i = getLayerAmount() - 1; i > 0; i--)
	{
		layers[i]->backPropagate(learnSpeed);
	}
}

//训练一批数据，输出步数和误差
void NeuralNet::train(int times, double tol)
{
	int a = realDataAmount;
	//批量学习时，节点数据量等于实际数据量
	if (learnMode == Online)
	{
		a = 1;
	}

	auto output = new double[outputAmount*realDataAmount];

	setInputData(inputData, inputAmount, realDataAmount);
	setExpectData(expectData, outputAmount, realDataAmount);

	for (int count = 0; count < times; count++)
	{
 		if (learnMode == Online)
 		{
 			for (int i = 0; i < realDataAmount; i++)
 			{
 				learn();
 			}
 		}
		else
		{
			learn();
		}

		//计算误差
		if (count % 1000 == 0)
		{
			double e = getLastLayer()->delta->ddot() / (realDataAmount*outputAmount);
			double e1 = 0;
			activeOutputValue(inputData, output, realDataAmount);
			for (int i = 0; i < realDataAmount; i++)
			{
				for (int j = 0; j < outputAmount; j++)
				{
					double e2 = output[i*outputAmount + j] - expectData[i*outputAmount + j];
					e1 += e2*e2;
				}
			}
			e1 = e1 / (realDataAmount*outputAmount);
			fprintf(stdout, "step = %d,\tmean square error = %f, %f\n", count, e, e1);
			if (e < tol) break;
		}		
	}
	delete [] output;
}

//读取数据
//这里的处理可能不是很好
void NeuralNet::readData(const char* filename)
{
	int mark = 2;
	//数据格式：前两个是输入变量数和输出变量数，之后依次是每组的输入和输出，是否有回车不重要
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	inputAmount = int(v[0]);
	outputAmount = int(v[1]);

	realDataAmount = (n - mark) / (inputAmount + outputAmount);
	inputData = new double[inputAmount * realDataAmount];
	expectData = new double[outputAmount * realDataAmount];

	//这里的写法太难看了
	int k = mark, k1 = 0, k2 = 0;

	for (int i_data = 1; i_data <= realDataAmount; i_data++)
	{
		for (int i = 1; i <= inputAmount; i++)
		{
			inputData[k1++] = v[k++];
		}
		for (int i = 1; i <= outputAmount; i++)
		{
			expectData[k2++] = v[k++];
		}
	}
	//测试用
	//realDataAmount = 10;
}

//输出键结值
void NeuralNet::outputBondWeight(const char* filename)
{
	FILE *fout = stdout;
	if (filename)
		fout = fopen(filename, "w+t");

	fprintf(fout,"\nNet information:\n", layers.size());
	fprintf(fout,"%d\tlayers\n", layers.size());
	for (int i_layer = 0; i_layer < layers.size(); i_layer++)
	{
		fprintf(fout,"layer %d has %d nodes\n", i_layer, layers[i_layer]->nodeAmount);
	}

	fprintf(fout,"---------------------------------------\n");
	for (int i_layer = 0; i_layer < layers.size() - 1; i_layer++)
	{
		auto& layer1 = layers[i_layer];
		auto& layer2 = layers[i_layer + 1];
	}
	if (filename)
		fclose(fout);
}

//依据输入数据创建神经网
//此处是具体的网络结构
void NeuralNet::createByData(NeuralLayerMode layerMode /*= HaveConstNode*/, int layerAmount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	this->createLayers(layerAmount);

	if (layerMode == HaveConstNode)
		getFirstLayer()->initData(inputAmount + 1, realDataAmount);
	else
		getFirstLayer()->initData(inputAmount, realDataAmount);
	getFirstLayer()->type = Input;

	for (int i = 1; i < layerAmount - 1; i++)
	{
		getLayer(i)->initData(nodesPerLayer, realDataAmount);
	}
	
	getLastLayer()->initData(outputAmount, realDataAmount);
	//getLastLayer()->setFunctions(ActiveFunctions::linear, ActiveFunctions::dlinear);
	getLastLayer()->initExpect();
	getLastLayer()->type = Output;

	for (int i = 1; i < layerAmount; i++)
	{
		layers[i]->connetPrevlayer(layers[i - 1]);
	}

	//printf("%d,%d,%d\n", layer->getNodeAmount(), layer->getNode(0)->bonds.size(), getLayer(1));
}

//依据键结值创建神经网
void NeuralNet::createByLoad(const char* filename, bool haveConstNode /*= true*/)
{
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	std::vector<int> v_int;
	v_int.resize(n);
	for (int i = 0; i < n; i++)
	{
		v_int[i] = int(v[i]);
	}
	int k = 0;	
	this->createLayers(v_int[k++]);
}


