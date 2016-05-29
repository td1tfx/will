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

//�������нڵ㵽һ��vector����
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

//����ѧϰģʽ
void NeuralNet::setLearnMode(NeuralNetLearnMode lm)
{
	learnMode = lm;
}

//�����񾭲�
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


//�����һ��������Ϊ�������ݣ�д����hack����
void NeuralNet::selectTest()
{
	//����ԭ��������
	auto input = new double[inputAmount*realDataAmount];
	auto output = new double[outputAmount*realDataAmount];
	memcpy(input, inputData, sizeof(double)*inputAmount*realDataAmount);
	memcpy(output, expectData, sizeof(double)*outputAmount*realDataAmount);

	inputTestData = new double[inputAmount*realDataAmount];
	expectTestData = new double[outputAmount*realDataAmount];

	isTest.resize(realDataAmount);
	int test = 0;
	int p = 0, p_data = 0, p_test = 0;
	int it = 0, id = 0;
	for (int i = 0; i < realDataAmount; i++)
	{
		//���ָ���˲�������������������ڲ��ԣ��������ѡ10%����
		if (this->testDataAmount == 0)
		{
			isTest[i] = (0.9 < 1.0*rand() / RAND_MAX);
		}
		else
		{
			isTest[i] = i >= realDataAmount - this->testDataAmount;
		}
		if (isTest[i])
		{
			memcpy(inputTestData + inputAmount*it, input+inputAmount*i, sizeof(double)*inputAmount);
			memcpy(expectTestData + outputAmount*it, output + outputAmount*i, sizeof(double)*outputAmount);
			test++;
			it++;
		}
		else
		{
			memcpy(inputData + inputAmount*id, input + inputAmount*i, sizeof(double)*inputAmount);
			memcpy(expectData + outputAmount*id, output + outputAmount*i, sizeof(double)*outputAmount);
			id++;
		}
	}
	realDataAmount -= test;
}

//�����ϵĽ���Ͳ��Լ��Ľ��
void NeuralNet::test()
{

	auto output_train = new double[outputAmount*realDataAmount];

	//���ȫ������
	setNodeDataAmount(realDataAmount);
	activeOutputValue(inputData, output_train, realDataAmount);
	fprintf(stdout, "\n%d groups train data comparing with expect (result -> expect):\n---------------------------------------\n", realDataAmount);
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
	delete output_train;

	if (testDataAmount <= 0) return;
	auto output_test = new double[outputAmount*testDataAmount];
	activeOutputValue(inputTestData, output_test, testDataAmount);
	fprintf(stdout, "\n%d groups test data (result -> expect):\n---------------------------------------\n", testDataAmount);
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
	delete output_test;
}

//�������
//���ﰴ��ǰ������Ӧ�����𲽻��ݼ��㣬ʹ��ջ��������˳�򣬴����ƺ��޸�
void NeuralNet::activeOutputValue(double* input, double* output, int amount)
{

	
	for (auto& node : getFirstLayer()->getNodeVector())
	{
		for (int i = 0; i < amount; i++)
		{
			//��������ڵ㣬��ǿ����������
			node->setOutput(input[i*inputAmount + node->id], i);
		}
	}


	if (activeMode == ByLayer)
	{
		//�ֲ㷴�򴫲�
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
		//������Ԫ�𲽻���
		for (auto& node : nodes)
		{
			node->actived = false;
		}
		std::vector<NeuralNode*> calstack;
		for (auto& node : getLastLayer()->getNodeVector())
		{
			calstack.push_back(node);
		}

		while (calstack.size() > 0)
		{
			auto node = calstack.back();
			bool all_prev_finished = true;
			for (auto& b : node->prevBonds)
			{
				if (b.second->startNode->actived == false)
				{
					all_prev_finished = false;
					calstack.push_back(b.second->startNode);
				}
			}
			if (all_prev_finished)
			{
				node->active();
				calstack.pop_back();
			}
		}
	}
	if (workMode == Classify)
		getLastLayer()->markMax();
	if (workMode == Probability)
		getLastLayer()->normalized();
	//��ѧϰ�׶ο��Բ����
	if (output)
	{
		for (auto& node : getLastLayer()->getNodeVector())
		{
			for (int i = 0; i < amount; i++)
			{
				output[i*outputAmount + node->id] = node->getOutput(i);
			}
		}
	}
}

//ѧϰ���ݣ�amount����1������ѧϰ��Ϊ1������ѧϰ����Ҫ����Ϊ����ֵ
//����Ҫ�ظ����ѧϰ��Ϊ�����Ч�ʣ������������������
void NeuralNet::learn(double* input, double* output, int amount)
{
	if (amount <= 0) return;
	if (amount > nodeDataAmount) amount = nodeDataAmount;

	activeOutputValue(input, nullptr, amount);

	//�����������
	//����ķ�ʽӦ�����𲽻��ݣ����ﴦ��ķ����Ƚϼ�
	auto layer_output = layers.back();
	int k = 0;
	for (int i = 0; i < amount; i++)
	{
		for (auto& node : layer_output->getNodeVector())
		{
			node->setExpect(output[i*outputAmount + node->id], i);
		}
	}

	if (backPropageteMode == ByLayer)
	{
		//���㷴�򴫲�
		for (int i_layer = layers.size() - 1; i_layer >= 0; i_layer--)
		{
			auto layer = layers[i_layer];
			for (auto& node : layer->getNodeVector())
			{
				node->backPropagate(learnSpeed);
			}
		}
	}
	else
	{
		//���ݼ���
		for (auto& node : nodes)
		{
			node->backPropageted = false;
		}
		std::vector<NeuralNode*> calstack;
		for (auto& node : getFirstLayer()->getNodeVector())
		{
			calstack.push_back(node);
		}

		while (calstack.size() > 0)
		{
			auto node = calstack.back();
			bool all_next_finished = true;
			for (auto& b : node->nextBonds)
			{
				if (b.second->endNode->backPropageted == false)
				{
					all_next_finished = false;
					calstack.push_back(b.second->endNode);
				}
			}
			if (all_next_finished)
			{
				node->backPropagate(learnSpeed);
				calstack.pop_back();
			}
		}
	}
}

//ѵ��һ�����ݣ�������������
void NeuralNet::train(int times, double tol)
{
	int a = realDataAmount;
	//����ѧϰʱ���ڵ�����������ʵ��������
	if (learnMode == Online)
		a = 1;
	setNodeDataAmount(a);

	double e = calTol();

	fprintf(stdout, "step = %d,\tmean square error = %f\n", 0, e);
	if (e < tol) return;

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

		//�������
		if (count % 1000 == 0)
		{
			double e = calTol();			
			fprintf(stdout, "step = %d,\tmean square error = %f\n", count, e);
			if (e < tol) break;
		}		
	}

}

double NeuralNet::calTol()
{
	double e = 0;
	auto output = new double[outputAmount*realDataAmount];
	setNodeDataAmount(realDataAmount);
	activeOutputValue(inputData, output, realDataAmount);
	if (learnMode == Online)
		setNodeDataAmount(1);
	for (int i = 0; i < realDataAmount; i++)
	{
		for (int j = 0; j < outputAmount; j++)
		{
			//double e1 = 1 - output[i*outputAmount + j] / expectData[i*outputAmount + j];
			double e1 = output[i*outputAmount + j] - expectData[i*outputAmount + j];
			e += e1*e1;
		}
	}
	e = e / (realDataAmount*outputAmount);
	delete[] output;
	return e;
}

//��ȡ����
//����Ĵ�����ܲ��Ǻܺ�
void NeuralNet::readData(const char* filename, double* input /*= nullptr*/, double* output /*= nullptr*/, int amount /*= -1*/)
{
	//���ݸ�ʽ��ǰ����������������������������֮��������ÿ��������������Ƿ��лس�����Ҫ
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	inputAmount = int(v[0]);
	outputAmount = int(v[1]);
	testDataAmount = int(v[2]);
	//����Ĭ�ϲ����Ĵ���
	if (amount == -1)
	{
		amount = (n - 3) / (inputAmount + outputAmount);
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

	int k = 3, k1 = 0, k2 = 0;
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
	//������
	//realDataAmount = 10;
}

//�������ֵ
void NeuralNet::outputBondWeight(const char* filename)
{
	FILE *fout = stdout;
	if (filename)
		fout = fopen(filename, "w+t");

	fprintf(fout, "\nNet information:\n", layers.size());
	fprintf(fout, "%d\tlayers\n", layers.size());
	for (int i_layer = 0; i_layer < layers.size(); i_layer++)
	{
		fprintf(fout, "layer %d has %d nodes\n", i_layer, layers[i_layer]->getNodeAmount());
	}
	//printf("start\tend\tweight\n");
	fprintf(fout, "---------------------------------------\n");
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
						fprintf(fout, " %d_%d\t%d_%d\t%14.11lf\n", i_layer, node1->id, i_layer + 1, node2->id, b.second->weight);
					}
				}
			}
		}
	}
	if (filename)
		fclose(fout);
}

//�����������ݴ�������
//�˴��Ǿ��������ṹ
void NeuralNet::createByData(NeuralLayerMode layerMode /*= HaveConstNode*/, int layerAmount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	this->createLayers(layerAmount);
	auto layer_input = layers.at(0);

	if (layerMode == HaveConstNode)
		layer_input->createNodes(inputAmount + 1, Input, layerMode);
	else
		layer_input->createNodes(inputAmount, Input, layerMode);
	auto layer_output = layers.back();
	layer_output->createNodes(outputAmount, Output);

	for (auto& node : layer_output->getNodeVector())
	{
		if (workMode == Fit)
			node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
		if (workMode == Probability)
			node->setFunctions(ActiveFunctions::exp1, ActiveFunctions::dexp1);
	}

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

//���ݼ���ֵ��������
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
	if (workMode == Probability)
	{
		for (auto node : getLastLayer()->nodes)
		{
			node->setFunctions(ActiveFunctions::exp1, ActiveFunctions::dexp1);
		}
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

//����ÿ���ڵ��������
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

