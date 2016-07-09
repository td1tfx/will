#include "NeuralNet.h"
#include "Random.h"

NeuralNet::NeuralNet()
{

}


NeuralNet::~NeuralNet()
{
	for (int i = 0; i < LayerCount; i++)
	{
		delete Layers[i];
	}
	delete[] Layers;
	if (train_input)
		delete train_input;
	if (train_expect)
		delete train_expect;
	if (test_input)
		delete test_input;
	if (test_expect)
		delete test_expect;
}

//���У�ע���ݴ�������
void NeuralNet::run(Option* op)
{
	BatchMode = NeuralNetLearnType(op->getInt("BatchMode"));
	MiniBatchCount = std::max(1, op->getInt("MiniBatch"));
	WorkType = NeuralNetWorkType(op->getInt("WorkMode"));

	LearnSpeed = op->getReal("LearnSpeed", 0.5);
	Lambda = op->getReal("Regular");

	MaxGroup = op->getInt("MaxGroup", 100000);
	
	if (op->getInt("UseMNIST") == 0)
	{
		readData(op->getString("TrainDataFile").c_str(), &train_groupCount, &train_input, &train_expect);
		readData(op->getString("TestDataFile").c_str(), &test_groupCount, &test_input, &test_expect);
	}
	else
	{
		readMNIST();
	}

	//�������ļ�ǿ�����´������磬��̫����
	//if (readStringFromFile(_option.LoadFile) == "")
	//	_option.LoadNet == 0;

	std::vector<int> v;
	int n = findNumbers(op->getString("NodePerLayer"), &v);

	if (op->getInt("LoadNet") == 0)
		createByData(op->getInt("Layer", 3), v[0]);
	else
		createByLoad(op->getString("LoadFile").c_str());

	//selectTest();
	train(op->getInt("TrainTimes", 1000), op->getInt("OutputInterval", 1000),
		op->getReal("Tol", 1e-3), op->getReal("Dtol", 0.0));
	if (op->getString("SaveFile") != "")
	{
		saveInfo(op->getString("SaveFile").c_str());
	}

	test(op->getInt("ForceOutput"), op->getInt("TestMax"));
	extraTest(op->getString("ExtraTestDataFile").c_str(),op->getInt("ForceOutput"), op->getInt("TestMax"));
}

//����ѧϰģʽ
void NeuralNet::setLearnType(NeuralNetLearnType lm, int lb /*= -1*/)
{
	BatchMode = lm;
	//����ѧϰʱ���ڵ�����������ʵ��������
	if (BatchMode == nl_Online)
	{
		MiniBatchCount = 1;
	}
	//�����������������
	if (BatchMode == nl_MiniBatch)
	{
		MiniBatchCount = lb;
	}
}

void NeuralNet::setWorkType(NeuralNetWorkType wm)
{
	WorkType = wm;
	if (wm == nw_Probability)
	{
		getLastLayer()->setActiveFunction(af_Softmax);
	}
	if (wm == nw_Classify)
	{
		getLastLayer()->setActiveFunction(af_Findmax);
	}
}

//�����񾭲�
void NeuralNet::createLayers(int layerCount)
{
	Layers = new NeuralLayer*[layerCount];
	LayerCount = layerCount;
	for (int i = 0; i < layerCount; i++)
	{
		auto layer = NeuralLayerFactory::createLayer(lc_Full);
		layer->Id = i;
		Layers[i] = layer;
	}
}


//ѵ��һ�����ݣ��������������ѵ������Ϊ0�������Ϊ������ģʽ
void NeuralNet::train(int times, int interval, real tol, real dtol)
{
	if (times <= 0) return;
	//��������ʼ��������㹻С�Ͳ�ѵ����
	//����������������������������𣬹���ʱ�״�ѵ��������������
	real e = 0;
	train_input->tryUploadToCuda();
	train_expect->tryUploadToCuda();
	active(train_input, train_expect, nullptr, train_groupCount, MiniBatchCount, false, &e);
	fprintf(stdout, "step = %e, mse = %e\n", 0.0, e);
	if (e < tol) return;
	real e0 = e;

	switch (BatchMode)
	{
	case nl_Whole:
		MiniBatchCount = resetGroupCount(train_groupCount);
		break;
	case nl_Online:
		resetGroupCount(1);
		MiniBatchCount = 1;
		break;
	case nl_MiniBatch:
		if (MiniBatchCount > 0)
			resetGroupCount(MiniBatchCount);
		break;
	default:
		break;
	}

	//ѵ������
	for (int count = 1; count <= times; count++)
	{
		//getFirstLayer()->step = count;
		active(train_input, train_expect, nullptr, train_groupCount, MiniBatchCount, true, count % interval == 0 ? &e : nullptr);
		if (count % interval == 0 || count == times)
		{
			fprintf(stdout, "step = %e, mse = %e, diff(mse) = %e\n", real(count), e, e0 - e);
			if (e < tol || std::abs(e - e0) < dtol) break;
			e0 = e;
		}
	}
}

void NeuralNet::active(Matrix* input, Matrix* expect, Matrix* output, int groupCount, int batchCount,
	bool learn /*= false*/, real* error /*= nullptr*/)
{
	Random<real> r;
	r.reset();
	if (error) *error = 0;
	for (int i = 0; i < groupCount; i += batchCount)
	{
		int selectgroup = i;
		if (batchCount <= 1)
		{
			selectgroup = int(r.rand_uniform()*groupCount);
		}
		int n = resetGroupCount(std::min(batchCount, groupCount - selectgroup));
		if (input)
		{
			getFirstLayer()->OutputMatrix->shareData(input, 0, selectgroup);
		}
		if (expect)
		{
			getLastLayer()->ExpectMatrix->shareData(expect, 0, selectgroup);
		}

		for (int i_layer = 1; i_layer < getLayerCount(); i_layer++)
		{
			Layers[i_layer]->activeOutput();
		}

		if (learn)
		{
			for (int i_layer = getLayerCount() - 1; i_layer > 0; i_layer--)
			{
				Layers[i_layer]->updateDelta();
				Layers[i_layer]->updateWeightBias(LearnSpeed, Lambda);
			}
		}
		if (output)
		{
			getOutputData(output, n, selectgroup);
		}
		//������ע������㷨����minibatch���ϸ�
		if (error)
		{
			if (!learn)
			{
				getLastLayer()->updateDelta();
			}
			*error += getLastLayer()->getDeltaMatrix()->ddot() / groupCount / OutputNodeCount;
		}
	}
}


void NeuralNet::getOutputData(Matrix* output, int groupCount, int col/*= 0*/)
{
	getLastLayer()->getOutputMatrix()->memcpyDataOut(output->getDataPointer(0, col), OutputNodeCount*groupCount);
}


//��ȡ����
//����Ĵ�����ܲ��Ǻܺ�
void NeuralNet::readData(const char* filename, int* count, Matrix** input, Matrix** expect)
{
	*count = 0;
	if (std::string(filename) == "") 
		return;

	int mark = 3;
	//���ݸ�ʽ��ǰ����������������������������֮��������ÿ��������������Ƿ��лس�����Ҫ
	std::string str = readStringFromFile(filename);
	if (str == "")
		return;
	std::vector<real> v;
	int n = findNumbers(str, &v);
	if (n <= 0) return;
	InputNodeCount = int(v[0]);
	OutputNodeCount = int(v[1]);

	*count = (n - mark) / (InputNodeCount + OutputNodeCount);
	*input = new Matrix(InputNodeCount, *count, md_Inside, mc_NoCuda);
	*expect = new Matrix(OutputNodeCount, *count, md_Inside, mc_NoCuda);

	//д��̫�ѿ���
	int k = mark, k1 = 0, k2 = 0;

	for (int i_data = 1; i_data <= (*count); i_data++)
	{
		for (int i = 1; i <= InputNodeCount; i++)
		{
			(*input)->getData(k1++) = v[k++];
		}
		for (int i = 1; i <= OutputNodeCount; i++)
		{
			(*expect)->getData(k2++) = v[k++];
		}
	}
}

int NeuralNet::resetGroupCount(int n)
{
	if (n == NeuralLayer::GroupCount) return n;
	if (n > MaxGroup)
		n = MaxGroup;
	NeuralLayer::setGroupCount(n);
	for (int i = 0; i < LayerCount; i++)
	{
		Layers[i]->resetGroupCount();
	}
	return n;
}

//�����������ݴ�������������Ľڵ���ֻ�����ز�����
//�˴��Ǿ��������ṹ
void NeuralNet::createByData(int layerCount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	NeuralLayer::setGroupCount(MiniBatchCount);

	this->createLayers(layerCount);

	getFirstLayer()->initData(lt_Input, InputNodeCount, 0);
	fprintf(stdout, "Layer %d has %d nodes.\n", 0, InputNodeCount);
	for (int i = 1; i < layerCount - 1; i++)
	{
		getLayer(i)->initData(lt_Hidden, nodesPerLayer, 0);
		fprintf(stdout, "Layer %d has %d nodes.\n", i, nodesPerLayer);
	}
	getLastLayer()->initData(lt_Output, OutputNodeCount, 0);
	fprintf(stdout, "Layer %d has %d nodes.\n", layerCount - 1, OutputNodeCount);

	for (int i = 1; i < layerCount; i++)
	{
		Layers[i]->connetPrevlayer(Layers[i - 1]);
	}
}

//�������ֵ
void NeuralNet::saveInfo(const char* filename)
{
	FILE *fout = stdout;
	if (filename)
		fout = fopen(filename, "w+t");
	if (!fout)
	{
		fprintf(stderr, "Can not open file %s\n", filename);
		return;
	}

	fprintf(fout, "Net information:\n");
	fprintf(fout, "%d\tlayers\n", LayerCount);
	for (int i_layer = 0; i_layer < getLayerCount(); i_layer++)
	{
		fprintf(fout, "layer %d has %d nodes\n", i_layer, Layers[i_layer]->OutputCountPerGroup);
	}

	fprintf(fout, "---------------------------------------\n");
	for (int i_layer = 1; i_layer < getLayerCount(); i_layer++)
	{
		Layers[i_layer]->saveInfo(fout);
	}

	if (filename)
		fclose(fout);
}

//���ݼ���ֵ��������
void NeuralNet::createByLoad(const char* filename)
{
	std::string str = readStringFromFile(filename);
	if (str == "")
		return;
	std::vector<real> vv;
	int n = findNumbers(str, &vv);
	auto v = new real[n];
	for (int i = 0; i < n; i++)
		v[i] = vv[i];

	int k = 0;
	int layerCount = int(v[k++]);
	this->createLayers(layerCount);
	getFirstLayer()->Type = lt_Input;
	getLastLayer()->Type = lt_Output;
	k++;
	for (int i_layer = 0; i_layer < layerCount; i_layer++)
	{
		getLayer(i_layer)->initData(getLayer(i_layer)->Type, int(v[k]), 0);
		fprintf(stdout, "Layer %d has %d nodes.\n", i_layer, int(v[k]));
		k += 2;
	}
	k = 1 + layerCount * 2;
	for (int i_layer = 0; i_layer < layerCount - 1; i_layer++)
	{
		auto& layer1 = Layers[i_layer];
		auto& layer2 = Layers[i_layer + 1];
		layer2->connetPrevlayer(layer1);
		int readcount = layer2->loadInfo(v + k, n - k);
		k += readcount;
	}
	delete[] v;
}

void NeuralNet::readMNIST()
{
	InputNodeCount = 784;
	OutputNodeCount = 10;

	train_groupCount = 60000;
	train_input = new Matrix(InputNodeCount, train_groupCount, md_Inside, mc_NoCuda);
	train_expect = new Matrix(OutputNodeCount, train_groupCount, md_Inside, mc_NoCuda);

	test_groupCount = 10000;
	test_input = new Matrix(InputNodeCount, train_groupCount, md_Inside, mc_NoCuda);
	test_expect = new Matrix(OutputNodeCount, train_groupCount, md_Inside, mc_NoCuda);

	Test::MNIST_readImageFile("train-images.idx3-ubyte", train_input->getDataPointer());
	Test::MNIST_readLabelFile("train-labels.idx1-ubyte", train_expect->getDataPointer());

	Test::MNIST_readImageFile("t10k-images.idx3-ubyte", test_input->getDataPointer());
	Test::MNIST_readLabelFile("t10k-labels.idx1-ubyte", test_expect->getDataPointer());
}


void NeuralNet::selectTest()
{

}


//�����ϵĽ���Ͳ��Լ��Ľ��
void NeuralNet::test(int forceOutput /*= 0*/, int testMax /*= 0*/)
{
	outputTest("train", OutputNodeCount, train_groupCount, train_input, train_expect, forceOutput, testMax);
	outputTest("test", OutputNodeCount, test_groupCount, test_input, test_expect, forceOutput, testMax);
}

void NeuralNet::extraTest(const char* filename, int forceOutput /*= 0*/, int testMax /*= 0*/)
{
	int count = 0;
	Matrix *input, *expect;
	readData(filename, &count, &input, &expect);
	outputTest("extra test", OutputNodeCount, count, input, expect, forceOutput, testMax);
	delete input;
	delete expect;
}

void NeuralNet::outputTest(const char* info, int nodeCount, int groupCount, Matrix* input, Matrix* expect, int forceOutput, int testMax)
{
	if (groupCount <= 0) return;

	expect->tryDownloadFromCuda();
	auto output = new Matrix(nodeCount, groupCount, md_Inside, mc_NoCuda);
	input->tryUploadToCuda();
	active(input, nullptr, output, groupCount, resetGroupCount(groupCount));

	fprintf(stdout, "\n%d groups %s data:\n---------------------------------------\n", groupCount, info);
	if (forceOutput || groupCount <= 100)
	{
		for (int i = 0; i < groupCount; i++)
		{
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4f ", output->getData(j, i));
			}
			fprintf(stdout, " --> ");
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4f ", expect->getData(j, i));
			}
			fprintf(stdout, "\n");
		}
	}
	if (testMax)
	{
		auto outputMax = new Matrix(nodeCount, groupCount, md_Inside, mc_NoCuda);
		Matrix::activeForward(af_Findmax, output, outputMax);

		if (forceOutput || groupCount <= 100)
		{
			for (int i = 0; i < groupCount; i++)
			{
				int o = outputMax->indexColMaxAbs(i);
				int e = expect->indexColMaxAbs(i);
				fprintf(stdout, "%3d (%6.4f) --> %3d\n", o, output->getData(o, i), e);
			}
		}

		real n = 0;
		Matrix::minus(outputMax, expect, outputMax);
		n = outputMax->sumAbs() / 2;
		delete outputMax;
		fprintf(stdout, "Error of max value position: %d, %5.2f%%\n", int(n), n / groupCount * 100);
	}
	delete output;
}
