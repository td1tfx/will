#include "NeuralNet.h"



NeuralNet::NeuralNet()
{
}


NeuralNet::~NeuralNet()
{
	for (auto& layer : this->layers)
	{
		delete layer;
	}
	if (inputData) delete inputData;
	if (expectData)	delete expectData;
	if (inputTestData) delete inputTestData;
	if (expectTestData) delete expectTestData;
}

void NeuralNet::createLayers(int layerNumber)
{
	for (int i = 1; i <= layerNumber; i++)
	{
		auto layer = new NeuralLayer();
		layers.push_back(layer);
	}
}

void NeuralNet::learn(double* input, double* output)
{
	auto output_real = new double[outputNodeAmount*dataAmount];
	activeOutputValue(input, output_real);

	//这里是输出层
	//正规的方式应该是逐步回溯，这里处理的方法比较简单
	auto layer_output = layers.back();
	int k = 0;
	for (int j = 0; j < dataAmount; j++)
	{
		for (int i = 0; i < outputNodeAmount; i++)
		{
			auto& node = layer_output->getNode(i);
			node->setExpect(output[k++], j);
		}
	}

	for (int l = layers.size() - 1; l >= 0; l--)
	{
		auto layer = layers[l];
		for (int j = 0; j < layer->nodes.size(); j++)
		{
			auto node = layer->getNode(j);
			node->updateDelta();
			for (auto b : node->prevBonds)
			{
				auto& bond = b.second;
				bond->updateWeight(learnSpeed);
			}
		}
	}
	delete output_real;
}

void NeuralNet::train(int times, double tol)
{
	for (int count = 0; count < times; count++)
	{
		learn(inputData, expectData);
		double s = 0;
		auto& o = layers.back()->getNode(0)->outputValues;
		if (count % 1000 == 0)
		{
			for (int i = 0; i < dataAmount; i++)
			{
				//printf("%f\t", output_real[i]/ output[i]-1);
				double s1 = 1 - o[i] / expectData[i];
				s += s1*s1;
			}
			fprintf(stdout, "%d, %f, %f\n", count, s, s / dataAmount);
			if (s / dataAmount < tol) break;
		}
	}
}

//这里拆一部分数据为测试数据，写法有hack性质
void NeuralNet::selectTest()
{
	//备份原来的数据
	auto input = new double[inputNodeAmount*dataAmount];
	auto output = new double[outputNodeAmount*dataAmount];
	memcpy(input, inputData, sizeof(double)*inputNodeAmount*dataAmount);
	memcpy(output, expectData, sizeof(double)*outputNodeAmount*dataAmount);

	inputTestData = new double[inputNodeAmount*dataAmount];
	expectTestData = new double[outputNodeAmount*dataAmount];

	isTest.resize(dataAmount);
	testDataAmount = 0;
	int p = 0, p_data = 0, p_test = 0;
	int it = 0, id = 0;
	for (int i = 0; i < dataAmount; i++)
	{
		isTest[i] = (0.9 < 1.0*rand() / RAND_MAX);
		if (isTest[i])
		{
			memcpy(inputTestData + inputNodeAmount*it, input+inputNodeAmount*i, sizeof(double)*inputNodeAmount);
			memcpy(expectTestData + outputNodeAmount*it, output + outputNodeAmount*i, sizeof(double)*outputNodeAmount);
			testDataAmount++;
			it++;
		}
		else
		{
			memcpy(inputData + inputNodeAmount*id, input + inputNodeAmount*i, sizeof(double)*inputNodeAmount);
			memcpy(expectData + outputNodeAmount*id, output + outputNodeAmount*i, sizeof(double)*outputNodeAmount);
			id++;
		}
	}
	dataAmount -= testDataAmount;
}

void NeuralNet::test()
{

	auto output_train = new double[outputNodeAmount*dataAmount];

	activeOutputValue(inputData, output_train);
	printf("\nTrain data comparing with expection:\n---------------------------------------\n");
	for (int i = 0; i < dataAmount; i++)
	{
		printf("%14.12lf\t%14.12lf\n", output_train[i], expectData[i]);
	}
	delete output_train;

	auto output_test = new double[outputNodeAmount*testDataAmount];
	activeOutputValue(inputTestData, output_test, testDataAmount);
	printf("\nTest data:\n---------------------------------------\n");
	for (int i = 0; i < testDataAmount; i++)
	{
		printf("%14.12lf\t%14.12lf\n", output_test[i], expectTestData[i]);
		//for (int j = 0; j < inputNodeAmount; j++)
		//	printf("%14.12lf\t", inputTestData[i*inputNodeAmount + j]);
		//for (int j = 0; j < outputNodeAmount; j++)
		//	printf("%14.12lf\t", outputTestData[i*outputNodeAmount + j]);
		//printf("\n");
	}
	delete output_test;
}

//计算输出
//这里按照前面的设计应该是逐步回溯计算，使用栈保存计算的顺序，待完善后修改
void NeuralNet::activeOutputValue(double* input, double* output, int amount /*= -1*/)
{
	if (amount < 0) amount = dataAmount;
	for (int i = 0; i < inputNodeAmount; i++)
	{
		for (int j = 0; j < amount; j++)
		{
			getLayer(0)->getNode(i)->outputValues[j] = input[j*inputNodeAmount + i];
		}
	}

	for (int l = 1; l < layers.size(); l++)
	{
		auto layer = layers[l];
		for (int n = 0; n < layer->nodes.size(); n++)
		{
			auto node = layer->getNode(n);
			node->collectInputValue();
			node->activeOutputValue();
		}
	}
	auto layer = layers.back();

	for (int i = 0; i < outputNodeAmount; i++)
	{
		for (int j = 0; j < amount; j++)
		{
			output[j*outputNodeAmount + i] = layer->getNode(i)->outputValues[j];
		}
	}
}


//这里的处理可能不是很好
void NeuralNet::readData(std::string& filename, double* input, double* output, int amount)
{
	//数据格式：前两个是输入变量数和输出变量数，之后依次是每组的输入和输出，是否有回车不重要
	std::string str = readStringFromFile(filename);
	str = str + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	inputNodeAmount = int(v[0]);
	outputNodeAmount = int(v[1]);
	
	//三个默认参数的处理
	if (amount == -1)
	{
		amount = (n - 2) / (inputNodeAmount + outputNodeAmount);
		dataAmount = amount;
	}
	if (input == nullptr)
	{
		input = new double[inputNodeAmount * amount];
		inputData = input;
	}
	if (output == nullptr)
	{
		output = new double[outputNodeAmount * amount];
		expectData = output;
	}	

	int k = 2, k1 = 0, k2 = 0;
	for (int i = 1; i <= amount; i++)
	{
		for (int j = 1; j <= inputNodeAmount; j++)
		{
			input[k1++] = v[k++];
		}
		for (int j = 1; j <= outputNodeAmount; j++)
		{
			output[k2++] = v[k++];
		}
	}
	//测试用
	//dataGroupAmount = 3;
}

//此处是具体的网络结构
void NeuralNet::setLayers(double learnSpeed, int layerAmount, bool haveConstNode)
{
	this->createLayers(layerAmount);
	auto layer_input = layers.at(0);

	if (haveConstNode)
		layer_input->createNodes(inputNodeAmount + 1, dataAmount, Input, true);
	else
		layer_input->createNodes(inputNodeAmount, dataAmount, Input, false);
	auto layer_output = layers.back();
	layer_output->createNodes(outputNodeAmount, dataAmount, Output);

	for (auto node : layer_output->nodes)
	{
		node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
	}

	//layers[1]->createNodes(34, dataGroupAmount);
	//layers[2]->createNodes(34, dataGroupAmount);
	for (int i = 1; i <= layerAmount - 2; i++)
	{
		auto layer = layers[i];
		layer->createNodes(50, dataAmount, Hidden);
		for (auto node : layer->nodes)
		{
			node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
		}
	}

	for (int i = 1; i < layers.size(); i++)
	{		
		layers[i]->connetPrevlayer(layers[i - 1]);
	}
	//printf("%d,%d,%d\n", layer->getNodeAmount(), layer->getNode(0)->bonds.size(), getLayer(1));
}

void NeuralNet::outputWeight()
{
	printf("\nNet information:\n", layers.size());
	printf("%d\tlayers\n", layers.size());
	for (int i = 0; i < layers.size(); i++)
	{
		printf("layer %d has %d nodes\n", i, layers[i]->getNodeAmount());
	}
	//printf("start\tend\tweight\n");
	printf("---------------------------------------\n");
	for (int i = 0; i < layers.size() - 1; i++)
	{
		auto& l1 = layers[i];
		auto& l2 = layers[i+1];
		for (int j1 = 0; j1 < l1->getNodeAmount(); j1++)
		{
			auto& n1 = l1->getNode(j1);
			for (int j2 = 0; j2 < l2->getNodeAmount(); j2++)
			{
				auto& n2 = l2->getNode(j2);
				for (auto& b : n1->nextBonds)
				{
					auto& bond = b.second;
					if (n1==bond->startNode&&n2==bond->endNode)
					{
						printf(" %d_%d\t%d_%d\t%14.11lf\n", i, j1, i + 1, j2, b.second->weight);
					}
				}
			}
		}
	}
}

void NeuralNet::createByFile(std::string& filename)
{

}
