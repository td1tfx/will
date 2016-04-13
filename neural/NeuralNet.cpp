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
	//����ķ�ʽӦ�����𲽻��ݣ����ﴦ��ķ����Ƚϼ�
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

	for (int j = 0; j < layer_output->nodes.size(); j++)
	{
		auto node = layer_output->getNode(j);
		node->updateDelta();
		for (auto b : node->prevBonds)
		{
			auto& bond = b.second;
			bond->updateWeight(learnSpeed);
		}
	}
	for (int l = layers.size() - 2; l >= 0; l--)
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

void NeuralNet::train()
{
	for (int j = 1; j <= 10000000; j++)
	{
		learn(inputData, outputData);
		double s = 0;
		auto& o = layers.back()->getNode(0)->outputValues;
		if (j % 1000 == 0) 
		{
			for (int i = 0; i < dataGroupAmount; i++)
			{
				//printf("%f\t", output_real[i]/ output[i]-1);
				double s1 = o[i] - outputData[i];
				s += s1*s1;
			}
			printf("%f\n", s);
		}
	}
}

void NeuralNet::test()
{
	auto o = new double[outputNodeAmount*dataGroupAmount];
	activeOutputValue(inputData, o);
	for (int i = 0; i < dataGroupAmount; i++)
	{
		//printf("%lf\t%lf\t%14.12lf\t%14.12lf\n", inputData[i*2], inputData[i*2+1], o[i], outputData[i]);
	}
}

//�������
//���ﰴ��ǰ������Ӧ�����𲽻��ݼ��㣬ʹ��ջ��������˳�򣬴����ƺ��޸�
void NeuralNet::activeOutputValue(double* input, double* output)
{
	for (int i = 0; i < inputNodeAmount; i++)
	{
		for (int j = 0; j < dataGroupAmount; j++)
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

//����Ĵ�����ܲ��Ǻܺ�
void NeuralNet::readData(std::string& filename)
{
	//���ݸ�ʽ��ǰ����������������������������֮��������ÿ��������������Ƿ��лس�����Ҫ
	std::string str = readStringFromFile(filename);
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
	//dataGroupAmount = 4;
}

//�˴��Ǿ��������ṹ
void NeuralNet::setLayers()
{
	learnSpeed = 0.5;
	int nl = 4;
	this->createLayers(nl);
	auto layer_input = layers.at(0);
	layer_input->createNodes(inputNodeAmount, dataGroupAmount, NeuralNodeType::Input);

	auto layer_output = layers.back();
	layer_output->createNodes(outputNodeAmount, dataGroupAmount, NeuralNodeType::Output);

	for (auto node : layer_output->nodes)
	{
		node->setFunctions(ActiveFunctions::sigmoid, ActiveFunctions::dsigmoid);
	}

	for (int i = 1; i <= nl-2; i++)
	{
		auto layer = layers.at(i);
		layer->createNodes(50, dataGroupAmount);
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