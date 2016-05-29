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

void NeuralNet::setWorkMode(NeuralNetWorkMode wm)
{
	 workMode = wm; 
	 if (wm == Probability)
	 {
		 getLastLayer()->setFunctions(ActiveFunctions::exp1, ActiveFunctions::dexp1);
	 }
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
			fprintf(stdout, "%8.4lf ", output_train[i*outputAmount + j]);
		}
		fprintf(stdout, " --> ");
		for (int j = 0; j < outputAmount; j++)
		{
			fprintf(stdout, "%8.4lf ", expectData[i*outputAmount + j]);
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
			fprintf(stdout, "%8.4lf ", output_test[i*outputAmount + j]);
		}
		fprintf(stdout, " --> ");
		for (int j = 0; j < outputAmount; j++)
		{
			fprintf(stdout, "%8.4lf ", expectTestData[i*outputAmount + j]);
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

	if (workMode == Probability)
	{
		getLastLayer()->normalized();
	}
	else if (workMode == Classify)
	{
		getLastLayer()->markMax();
	}

	if (output)
	{
		getOutputData(output, outputAmount, amount);
	}
}

void NeuralNet::setInputData(double* input, int nodeAmount, int groupAmount)
{
	getFirstLayer()->output->memcpyDataIn(input, sizeof(double)*nodeAmount*groupAmount);
}

void NeuralNet::getOutputData(double* output, int nodeAmount, int groupAmount)
{
	getLastLayer()->output->memcpyDataOut(output, sizeof(double)*nodeAmount*groupAmount);
}

void NeuralNet::setExpectData(double* expect, int nodeAmount, int groupAmount)
{
	getLastLayer()->expect->memcpyDataIn(expect, sizeof(double)*nodeAmount*groupAmount);
}

//学习过程
void NeuralNet::learn()
{
	//正向计算
	activeOutputValue(nullptr, nullptr, realDataAmount);
	//反向传播
	for (int i = getLayerAmount() - 1; i > 0; i--)
	{
		layers[i]->backPropagate(learnSpeed, lambda);
	}
}

//训练一批数据，输出步数和误差
void NeuralNet::train(int times /*= 1000000*/, int interval /*= 1000*/, double tol /*= 1e-3*/, double dtol /*= 1e-9*/)
{
	int a = realDataAmount;
	//批量学习时，节点数据量等于实际数据量
	if (learnMode == Online)
	{
		a = 1;
	}

	setInputData(inputData, inputAmount, realDataAmount);
	setExpectData(expectData, outputAmount, realDataAmount);

	//这里计算初始的误差，如果足够小就不训练了
	activeOutputValue(nullptr, nullptr, realDataAmount);
	getLastLayer()->updateDelta();
	double e = getLastLayer()->delta->ddot() / (realDataAmount*outputAmount);
	fprintf(stdout, "step = %e, mse = %e\n", 0.0, e);
	double e0 = e;
	if (e < tol) return;

	//训练过程
	for (int count = 1; count <= times; count++)
	{
		//getFirstLayer()->step = count;
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
		if (count % interval == 0)
		{
			e = getLastLayer()->delta->ddot() / (realDataAmount*outputAmount);
			fprintf(stdout, "step = %e, mse = %e, diff(e) = %e\n", double(count), e, e0 - e);
			if (e < tol || abs(e - e0) < dtol) break;
			e0 = e;
		}
	}
}

//读取数据
//这里的处理可能不是很好
void NeuralNet::readData(const char* filename)
{
	int mark = 3;
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

	//写法太难看了
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

	fprintf(fout,"\nNet information:\n");
	fprintf(fout,"%d\tlayers\n", layers.size());
	for (int i_layer = 0; i_layer < getLayerAmount(); i_layer++)
	{
		fprintf(fout,"layer %d has %d nodes\n", i_layer, layers[i_layer]->nodeAmount);
	}

	fprintf(fout,"---------------------------------------\n");
	for (int i_layer = 0; i_layer < getLayerAmount() - 1; i_layer++)
	{
		auto& layer1 = layers[i_layer];
		auto& layer2 = layers[i_layer + 1];
		fprintf(fout, "weight for layer %d to %d\n", i_layer + 1, i_layer);
		for (int i2 = 0; i2 < layer2->nodeAmount; i2++)
		{
			for (int i1 = 0; i1 < layer1->nodeAmount; i1++)
			{
				fprintf(fout, "%14.11lf ", layer2->weight->getData(i2, i1));
			}
			fprintf(fout, "\n");
		}
		fprintf(fout, "bias for layer %d\n", i_layer + 1);
		for (int i2 = 0; i2 < layer2->nodeAmount; i2++)
		{
			fprintf(fout, "%14.11lf ", layer2->bias->getData(i2));
		}
		fprintf(fout, "\n");
	}

	if (filename)
		fclose(fout);
}

//依据输入数据创建神经网
//此处是具体的网络结构
void NeuralNet::createByData(int layerAmount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	this->createLayers(layerAmount);

	getFirstLayer()->type = Input;
	getFirstLayer()->initData(inputAmount, realDataAmount);


	for (int i = 1; i < layerAmount - 1; i++)
	{
		getLayer(i)->initData(nodesPerLayer, realDataAmount);
	}
	
	getLastLayer()->type = Output;
	getLastLayer()->initData(outputAmount, realDataAmount);
	//getLastLayer()->setFunctions(ActiveFunctions::linear, ActiveFunctions::dlinear);


	for (int i = 1; i < layerAmount; i++)
	{
		layers[i]->connetPrevlayer(layers[i - 1]);
	}
}

//依据键结值创建神经网
void NeuralNet::createByLoad(const char* filename)
{
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	// 	for (int i = 0; i < n; i++)
	// 		printf("%14.11lf\n",v[i]);
	// 	printf("\n");
	std::vector<int> v_int;
	v_int.resize(n);
	for (int i_layer = 0; i_layer < n; i_layer++)
	{
		v_int[i_layer] = int(v[i_layer]);
	}
	int k = 0;
	int layerAmount = v_int[k++];
	this->createLayers(layerAmount);
	getFirstLayer()->type = Input;
	getLastLayer()->type = Output;
	k++;
	for (int i_layer = 0; i_layer < layerAmount; i_layer++)
	{
		getLayer(i_layer)->initData(v_int[k], realDataAmount);
		k += 2;
	}
	k = 1 + layerAmount * 2;
	for (int i_layer = 0; i_layer < layerAmount - 1; i_layer++)
	{
		auto& layer1 = layers[i_layer];
		auto& layer2 = layers[i_layer + 1];
		layer2->connetPrevlayer(layer1);
		k += 2;
		for (int i2 = 0; i2 < layer2->nodeAmount; i2++)
		{
			for (int i1 = 0; i1 < layer1->nodeAmount; i1++)
			{
				layer2->weight->getData(i2, i1) = v[k++];
			}
		}
		k += 1;
		for (int i2 = 0; i2 < layer2->nodeAmount; i2++)
		{
			layer2->bias->getData(i2) = v[k++];
		}
	}
}

void NeuralNet::readMNIST()
{
	inputAmount = MNISTFunctions::readImageFile("train-images.idx3-ubyte", inputData);
	outputAmount = MNISTFunctions::readLabelFile("train-labels.idx1-ubyte", expectData);
	realDataAmount = 1000;
}

