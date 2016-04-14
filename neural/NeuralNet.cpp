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
	if (inputData)
	{
		delete inputData;
	}
	if (outputData)
	{
		delete outputData;
	}
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
	auto output_real = new double[outputNodeAmount*dataGroupAmount];
	activeOutputValue(input, output_real);

	//�����������
	//����ķ�ʽӦ�����𲽻��ݣ����ﴦ���ķ����Ƚϼ�
	auto layer_output = layers.back();
	int k = 0;
	for (int j = 0; j < dataGroupAmount; j++)
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
		learn(inputData, outputData);
		double s = 0;
		auto& o = layers.back()->getNode(0)->outputValues;
		if (count % 1000 == 0)
		{
			for (int i = 0; i < dataGroupAmount; i++)
			{
				//printf("%f\t", output_real[i]/ output[i]-1);
				double s1 = 1 - o[i] / outputData[i];
				s += s1*s1;
			}
			fprintf(stdout, "%d, %f, %f\n", count, s, s / dataGroupAmount);
			if (s / dataGroupAmount < tol) break;
		}
	}
}

void NeuralNet::test()
{
	auto o = new double[outputNodeAmount*dataGroupAmount];
	activeOutputValue(inputData, o);
	printf("\nresults:\n---------------------------------------\n");
	for (int i = 0; i < dataGroupAmount; i++)
	{
		printf("%14.12lf\t%14.12lf\n", o[i], outputData[i]);
	}
}

//�������
//���ﰴ��ǰ������Ӧ�����𲽻��ݼ��㣬ʹ��ջ��������˳�򣬴����ƺ��޸�
void NeuralNet::activeOutputValue(double* input, double* output, int amount /*= -1*/)
{
	if (amount < 0) amount = dataGroupAmount;
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
		for (int j = 0; j < dataGroupAmount; j++)
		{
			output[j*outputNodeAmount + i] = layer->getNode(i)->outputValues[j];
		}
	}
}


//����Ĵ������ܲ��Ǻܺ�
void NeuralNet::readData(std::string& filename)
{
	//���ݸ�ʽ��ǰ����������������������������֮��������ÿ��������������Ƿ��лس�����Ҫ
	std::string str = readStringFromFile(filename);
	str = str + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	inputNodeAmount = int(v[0]);
	outputNodeAmount = int(v[1]);

	dataGroupAmount = (n - 2) / (inputNodeAmount + outputNodeAmount);

	int k = 2;
	int k1 = 0, k2 = 0;
	inputData = new double[inputNodeAmount * dataGroupAmount];
	outputData = new double[outputNodeAmount * dataGroupAmount];
	for (int i = 1; i <= dataGroupAmount; i++)
	{
		for (int j = 1; j <= inputNodeAmount; j++)
		{
			inputData[k1++] = v[k++];
		}
		for (int j = 1; j <= outputNodeAmount; j++)
		{
			outputData[k2++] = v[k++];
		}
	}
	//������
	//dataGroupAmount = 3;
}

//�˴��Ǿ��������ṹ
void NeuralNet::setLayers()
{
	learnSpeed = 0.5;
	int nl = 3;
	this->createLayers(nl);
	auto layer_input = layers.at(0);
	layer_input->createNodes(inputNodeAmount+1, dataGroupAmount, Input, true);

	auto layer_output = layers.back();
	layer_output->createNodes(outputNodeAmount, dataGroupAmount, Output);

	for (auto node : layer_output->nodes)
	{
		node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
	}

	//layers[1]->createNodes(34, dataGroupAmount);
	//layers[2]->createNodes(34, dataGroupAmount);
	for (int i = 1; i <= nl - 2; i++)
	{
		auto layer = layers[i];
		layer->createNodes(7, dataGroupAmount, Hidden);
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