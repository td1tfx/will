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

//保存所有节点到一个vector里面
void NeuralNet::initNodes()
{
	for (auto& layer : this->getLayerVector())
	{
		for (auto& node : layer->getNodeVector())
		{
			nodes.push_back(node);
		}
	}
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
	setNodeDataAmount(realDataAmount);
	activeOutputValue(inputData, output_train, realDataAmount);
	printf("\n%d groups train data comparing with expection:\n---------------------------------------\n", realDataAmount);
	for (int i = 0; i < realDataAmount; i++)
	{
		printf("%14.10lf\t%14.10lf\n", output_train[i], expectData[i]);
	}
	delete output_train;

	if (testDataAmount <= 0) return;
	auto output_test = new double[outputAmount*testDataAmount];
	activeOutputValue(inputTestData, output_test, testDataAmount);
	printf("\n%d groups test data:\n---------------------------------------\n", testDataAmount);
	for (int i = 0; i < testDataAmount; i++)
	{
		printf("%14.10lf\t%14.10lf\n", output_test[i], expectTestData[i]);
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
void NeuralNet::activeOutputValue(double* input, double* output, int amount)
{
	for (auto& node : nodes)
	{
		node->actived = false;
	}
	
	for (auto& node : getFirstLayer()->getNodeVector())
	{
		for (int i = 0; i < amount; i++)
		{
			//对于输入节点，是强制设置输入
			node->setOutput(input[i*inputAmount + node->id], i);
		}
	}


	if (activeMode == ByLayer)
	{
		for (int i = 1; i < layers.size(); i++)
		{
			for (auto& node : layers[i]->getNodeVector())
			{
				node->active();
			}
		}
	}
	else
	{
		std::vector<NeuralNode*> calstack;
		for (auto& node : getLastLayer()->getNodeVector())
		{
			calstack.push_back(node);
		}

		while (calstack.size() > 0)
		{
			auto node = calstack.back();
			bool all_prev_actived = true;
			for (auto& b : node->prevBonds)
			{
				if (b.second->startNode->actived == false)
				{
					all_prev_actived = false;
					calstack.push_back(b.second->startNode);
				}
			}
			if (all_prev_actived)
			{
				node->active();
				calstack.pop_back();
			}
		}
	}



	for (auto& node : getLastLayer()->getNodeVector())
	{
		for (int i = 0; i < amount; i++)
		{
			output[i*outputAmount + node->id] = node->getOutput(i);
		}
	}
}

//学习数据，amount大于1是批量学习，为1是在线学习，不要设置为其他值
//若需要重复多次学习，为了提高效率，最好事先设置数据量
void NeuralNet::learn(double* input, double* output, int amount)
{
	if (amount <= 0) return;
	if (amount > nodeDataAmount) amount = nodeDataAmount;

	auto output_real = new double[outputAmount*amount];
	activeOutputValue(input, output_real, amount);

	//这里是输出层
	//正规的方式应该是逐步回溯，这里处理的方法比较简单
	auto layer_output = layers.back();
	int k = 0;
	for (int i = 0; i < amount; i++)
	{
		for (auto& node : layer_output->getNodeVector())
		{
			node->setExpect(output[i*outputAmount + node->id], i);
		}
	}

	//反向传播
	for (int i_layer = layers.size() - 1; i_layer >= 0; i_layer--)
	{
		auto layer = layers[i_layer];
		for (auto& node : layer->getNodeVector())
		{
			node->BackPropagation(learnSpeed);
		}
	}
	delete output_real;
}

//训练一批数据，输出步数和误差
void NeuralNet::train(int times, double tol)
{
	int a = realDataAmount;
	//批量学习时，节点数据量等于实际数据量
	if (learnMode == Online)
		a = 1;
	setNodeDataAmount(a);

	auto output = new double[outputAmount*realDataAmount];

	for (int count = 0; count < times; count++)
	{
 		if (learnMode == Online)
 		{
 			for (int i = 0; i < realDataAmount; i++)
 			{
 				learn(inputData + inputAmount*i, expectData + outputAmount*i, 1);
 			}
 		}
		else
		{
			learn(inputData, expectData, a);
		}

		//计算误差
		if (count % 1000 == 0)
		{
			double e = 0;
			setNodeDataAmount(realDataAmount);
			activeOutputValue(inputData, output, realDataAmount);
			setNodeDataAmount(a);
			for (int i = 0; i < realDataAmount; i++)
			{
				//printf("%f\t", output_real[i]/ output[i]-1);
				double e1 = 1 - output[i] / expectData[i];
				e += e1*e1;
			}
			e = e / realDataAmount;
			fprintf(stdout, "step = %d,\tmean square error = %f\n", count, e);
			if (e < tol) break;
		}		
	}
	delete output;
}

//读取数据
//这里的处理可能不是很好
void NeuralNet::readData(const std::string& filename, double* input /*=nullptr*/, double* output /*= nullptr*/, int amount /*= -1*/)
{
	//数据格式：前两个是输入变量数和输出变量数，之后依次是每组的输入和输出，是否有回车不重要
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	inputAmount = int(v[0]);
	outputAmount = int(v[1]);
	
	//三个默认参数的处理
	if (amount == -1)
	{
		amount = (n - 2) / (inputAmount + outputAmount);
		realDataAmount = amount;
	}
	if (input == nullptr)
	{
		input = new double[inputAmount * amount];
		inputData = input;
	}
	if (output == nullptr)
	{
		output = new double[outputAmount * amount];
		expectData = output;
	}	

	int k = 2, k1 = 0, k2 = 0;
	for (int i_data = 1; i_data <= amount; i_data++)
	{
		for (int i = 1; i <= inputAmount; i++)
		{
			input[k1++] = v[k++];
		}
		for (int i = 1; i <= outputAmount; i++)
		{
			output[k2++] = v[k++];
		}
	}
	//测试用
	//dataGroupAmount = 3;
}

//输出键结值
void NeuralNet::outputBondWeight()
{
	printf("\nNet information:\n", layers.size());
	printf("%d\tlayers\n", layers.size());
	for (int i_layer = 0; i_layer < layers.size(); i_layer++)
	{
		printf("layer %d has %d nodes\n", i_layer, layers[i_layer]->getNodeAmount());
	}
	//printf("start\tend\tweight\n");
	printf("---------------------------------------\n");
	for (int i_layer = 0; i_layer < layers.size() - 1; i_layer++)
	{
		auto& layer1 = layers[i_layer];
		auto& layer2 = layers[i_layer + 1];
		for (auto& node1 : layer1->getNodeVector())
		{
			for (auto& node2 : layer2->getNodeVector())
			{
				for (auto& b : node1->nextBonds)
				{
					auto& bond = b.second;
					if (node1 == bond->startNode && node2 == bond->endNode)
					{
						printf(" %d_%d\t%d_%d\t%14.11lf\n", i_layer, node1->id, i_layer + 1, node2->id, b.second->weight);
					}
				}
			}
		}
	}
}

//依据输入数据创建神经网
//此处是具体的网络结构
void NeuralNet::createByData(bool haveConstNode /*= true*/, int layerAmount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	this->createLayers(layerAmount);
	auto layer_input = layers.at(0);

	if (haveConstNode)
		layer_input->createNodes(inputAmount + 1, Input, true);
	else
		layer_input->createNodes(inputAmount, Input, false);
	auto layer_output = layers.back();
	layer_output->createNodes(outputAmount, Output);

	for (auto& node : layer_output->getNodeVector())
	{
		node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
	}

	//layers[1]->createNodes(34, dataGroupAmount);
	//layers[2]->createNodes(34, dataGroupAmount);
	for (int i = 1; i <= layerAmount - 2; i++)
	{
		auto layer = layers[i];
		layer->createNodes(nodesPerLayer, Hidden);
		for (auto& node : layer->getNodeVector())
		{
			node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
		}
	}

	for (int i = 1; i < layers.size(); i++)
	{		
		layers[i]->connetPrevlayer(layers[i - 1]);
	}
	//printf("%d,%d,%d\n", layer->getNodeAmount(), layer->getNode(0)->bonds.size(), getLayer(1));
	initNodes();
}

//依据键结值创建神经网
void NeuralNet::createByLoad(const std::string& filename, bool haveConstNode /*= true*/)
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

	for (int i = 0; i < getLayerAmount(); i++)
	{
		NeuralNodeType t = Hidden;
		if (i == 0) t = Input;
		if (i == getLayerAmount() - 1) t = Output;
		getLayer(v_int[k])->createNodes(v_int[k + 1], t);
		for (auto node : getLayer(v_int[k])->nodes)
		{
			node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
		}
		k += 2;
	}
	for (; k < n; k += 5)
	{
		NeuralNode::connect(getLayer(v_int[k])->getNode(v_int[k + 1]), getLayer(v_int[k + 2])->getNode(v_int[k + 3]), v[k + 4]);
	}
	if (haveConstNode)
	{
		auto layer = getFirstLayer();
		layer->getNode(layer->getNodeAmount() - 1)->type = Const;
	}
	initNodes();
	//inputAmount = getLayer(0)->getNodeAmount();
	//outputAmount = getLayer(get)
}

//设置每个节点的数据量
void NeuralNet::setNodeDataAmount(int amount)
{
	nodeDataAmount = amount;
	for (auto& layer : this->getLayerVector())
	{
		for (auto& node : layer->getNodeVector())
		{
			node->setDataAmount(amount);
		}
	}
}

