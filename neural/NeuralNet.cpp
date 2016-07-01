#include "NeuralNet.h"


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
	if (_train_inputData)
		delete _train_inputData;
	if (_train_expectData)
		delete _train_expectData;
	if (_test_inputData)
		delete _test_inputData;
	if (_test_expectData)
		delete _test_expectData;
}

//���У�ע���ݴ�������
void NeuralNet::run()
{
	BatchMode = NeuralNetLearnType(_option.getInt("BatchMode"));
	MiniBatchCount = std::max(1, _option.getInt("MiniBatch"));
	WorkType = NeuralNetWorkType(_option.getInt("WorkMode"));

	LearnSpeed = _option.getDouble("LearnSpeed", 0.5);
	Lambda = _option.getDouble("Regular");

	MaxGroup = _option.getInt("MaxGroup", 1e5);

	if (_option.getInt("UseCUDA"))
	{
		Matrix::initCublas();
	}
	if (_option.getInt("UseMNIST") == 0)
	{
		if (_option.getString("TrainDataFile") != "")
		{
			readData(_option.getString("TrainDataFile").c_str(), da_Train);
		}
	}
	else
	{
		readMNIST();
	}

	//�������ļ�ǿ�����´������磬��̫����
	//if (readStringFromFile(_option.LoadFile) == "")
	//	_option.LoadNet == 0;

	std::vector<double> v;
	int n = findNumbers(_option.getString("NodePerLayer"), v);

	if (_option.getInt("LoadNet") == 0)
		createByData(_option.getInt("Layer", 3), int(v[0]));
	else
		createByLoad(_option.getString("LoadFile").c_str());

	//net->selectTest();
	train(_option.getInt("TrainTimes", 1000), _option.getInt("OutputInterval", 1000), 
		_option.getDouble("Tol", 1e-3), _option.getDouble("Dtol", 0));
	test();

	if (_option.getString("SaveFile") != "")
		saveInfo(_option.getString("SaveFile").c_str());
	if (_option.getString("TestDataFile") != "")
	{
		readData(_option.getString("TestDataFile").c_str(), da_Test);
		test();
	}
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


//�������������δ����
void NeuralNet::active(Matrix* input, Matrix* expect, Matrix* output, int groupCount, int batchCount,
	bool learn /*= false*/, double* error /*= nullptr*/)
{
	if (error) *error = 0;
	for (int i = 0; i < groupCount; i += batchCount)
	{
		int n = resetGroupCount(std::min(batchCount, groupCount - i));
		if (input)
		{
			getFirstLayer()->OutputMatrix->shareData(input, 0, i);
		}
		if (expect)
		{
			getLastLayer()->ExpectMatrix->shareData(expect, 0, i);
		}

		for (int i_layer = 1; i_layer < getLayerCount(); i_layer++)
		{
			Layers[i_layer]->activeOutputValue();
		}

		if (learn)
		{
			for (int i_layer = getLayerCount() - 1; i_layer > 0; i_layer--)
			{
				Layers[i_layer]->updateDelta();
				Layers[i_layer]->backPropagate(LearnSpeed, Lambda);
			}
		}
		if (output)
		{
			getOutputData(output, n, i);
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


void NeuralNet::getOutputData(Matrix* output, int groupCount, int col/*=0*/)
{
	getLastLayer()->getOutputMatrix()->memcpyDataOut(output->getDataPointer(0, col), OutputNodeCount*groupCount);
}


//ѵ��һ�����ݣ��������������ѵ������Ϊ0�������Ϊ������ģʽ
void NeuralNet::train(int times /*= 1000000*/, int interval /*= 1000*/, double tol /*= 1e-3*/, double dtol /*= 1e-9*/)
{
	if (times <= 0) return;
	//��������ʼ��������㹻С�Ͳ�ѵ����
	//����������������������������𣬹���ʱ�״�ѵ��������������
	double e = 0;
	_train_inputData->tryUploadToCuda();
	_train_expectData->tryUploadToCuda();
	active(_train_inputData, _train_expectData, nullptr, _train_groupCount, MiniBatchCount, false, &e);
	fprintf(stdout, "step = %e, mse = %e\n", 0.0, e);
	if (e < tol) return;
	double e0 = e;

	switch (BatchMode)
	{
	case nl_Whole:
		MiniBatchCount = resetGroupCount(_train_groupCount);
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
		active(_train_inputData, _train_expectData, nullptr, _train_groupCount, MiniBatchCount, true, count % interval == 0 ? &e : nullptr);
		if (count % interval == 0 || count == times)
		{
			fprintf(stdout, "step = %e, mse = %e, diff(mse) = %e\n", double(count), e, e0 - e);
			if (e < tol || std::abs(e - e0) < dtol) break;
			e0 = e;
		}
	}
}

//��ȡ����
//����Ĵ�����ܲ��Ǻܺ�
void NeuralNet::readData(const char* filename, DateType dm/*= Train*/)
{
	_train_groupCount = 0;
	_test_groupCount = 0;

	int mark = 3;
	//���ݸ�ʽ��ǰ����������������������������֮��������ÿ��������������Ƿ��лس�����Ҫ
	std::string str = readStringFromFile(filename);
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	if (n <= 0) return;
	InputNodeCount = int(v[0]);
	OutputNodeCount = int(v[1]);

	auto groupCount = &_train_groupCount;
	auto inputData = &_train_inputData;
	auto expectData = &_train_expectData;
	if (dm == da_Test)
	{
		groupCount = &_test_groupCount;
		inputData = &_test_inputData;
		expectData = &_test_expectData;
	}

	*groupCount = (n - mark) / (InputNodeCount + OutputNodeCount);
	*inputData = new Matrix(InputNodeCount, *groupCount, 1, 0);
	*expectData = new Matrix(OutputNodeCount, *groupCount, 1, 0);

	//д��̫�ѿ���
	int k = mark, k1 = 0, k2 = 0;

	for (int i_data = 1; i_data <= (*groupCount); i_data++)
	{
		for (int i = 1; i <= InputNodeCount; i++)
		{
			(*inputData)->getData(k1++) = v[k++];
		}
		for (int i = 1; i <= OutputNodeCount; i++)
		{
			(*expectData)->getData(k2++) = v[k++];
		}
	}
	// 	for (int i = 0; i < 784 * 22; i++)
	// 	{
	// 		if ((*inputData)[i] > 0.5)
	// 			printf("%2.1f ", (*inputData)[i]);
	// 		else
	// 		{
	// 			printf("    ");
	// 			(*inputData)[i] = 0;
	// 		}
	// 		if (i % 28 == 27)
	// 			printf("\n");
	// 	}
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
	std::vector<double> vv;
	int n = findNumbers(str, vv);
	auto v = new double[n];
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
}

void NeuralNet::readMNIST()
{
	InputNodeCount = 784;
	OutputNodeCount = 10;

	_train_groupCount = 60000;
	_train_inputData = new Matrix(InputNodeCount, _train_groupCount, 1, 0);
	_train_expectData = new Matrix(OutputNodeCount, _train_groupCount, 1, 0);

	_test_groupCount = 1000;
	_test_inputData = new Matrix(InputNodeCount, _train_groupCount, 1, 0);
	_test_expectData = new Matrix(OutputNodeCount, _train_groupCount, 1, 0);

	MNISTFunctions::readImageFile("train-images.idx3-ubyte", _train_inputData->getDataPointer());
	MNISTFunctions::readLabelFile("train-labels.idx1-ubyte", _train_expectData->getDataPointer());

	MNISTFunctions::readImageFile("t10k-images.idx3-ubyte", _test_inputData->getDataPointer());
	MNISTFunctions::readLabelFile("t10k-labels.idx1-ubyte", _test_expectData->getDataPointer());
}

void NeuralNet::loadOption(const char* filename)
{
	_option.loadIni(filename);
}


//�����һ��������Ϊ�������ݣ�д����hack����
void NeuralNet::selectTest()
{
	/*
	//����ԭ��������
	auto input = new double[InputNodeCount*_train_groupCount];
	auto output = new double[OutputNodeCount*_train_groupCount];
	memcpy(input, _train_inputData, sizeof(double)*InputNodeCount*_train_groupCount);
	memcpy(output, _train_expectData, sizeof(double)*OutputNodeCount*_train_groupCount);

	_test_inputData = new double[InputNodeCount*_train_groupCount];
	_test_expectData = new double[OutputNodeCount*_train_groupCount];

	std::vector<bool> isTest;
	isTest.resize(_train_groupCount);

	_test_groupCount = 0;
	int p = 0, p_data = 0, p_test = 0;
	int it = 0, id = 0;
	for (int i = 0; i < _train_groupCount; i++)
	{
		isTest[i] = (0.9 < 1.0*rand() / RAND_MAX);
		if (isTest[i])
		{
			memcpy(_test_inputData + InputNodeCount*it, input + InputNodeCount*i, sizeof(double)*InputNodeCount);
			memcpy(_test_expectData + OutputNodeCount*it, output + OutputNodeCount*i, sizeof(double)*OutputNodeCount);
			_test_groupCount++;
			it++;
		}
		else
		{
			memcpy(_train_inputData + InputNodeCount*id, input + InputNodeCount*i, sizeof(double)*InputNodeCount);
			memcpy(_train_expectData + OutputNodeCount*id, output + OutputNodeCount*i, sizeof(double)*OutputNodeCount);
			id++;
		}
	}
	_train_groupCount -= _test_groupCount;
	resetGroupCount(_train_groupCount);
	*/
}

//�����ϵĽ���Ͳ��Լ��Ľ��
void NeuralNet::test()
{
	_train_expectData->tryDownloadFromCuda();
	if (_train_groupCount > 0)
	{
		auto train_output = new Matrix(OutputNodeCount, _train_groupCount, 1, 0);
		_train_inputData->tryUploadToCuda();
		active(_train_inputData, nullptr, train_output, _train_groupCount, resetGroupCount(_train_groupCount));
		fprintf(stdout, "\n%d groups train data:\n---------------------------------------\n", _train_groupCount);
		printResult(OutputNodeCount, _train_groupCount, train_output, _train_expectData);
		delete train_output;
	}
	if (_test_groupCount > 0)
	{
		auto test_output = new Matrix(OutputNodeCount, _test_groupCount, 1, 0);
		_test_inputData->tryUploadToCuda();
		active(_test_inputData, nullptr, test_output, _test_groupCount, resetGroupCount(_test_groupCount));
		fprintf(stdout, "\n%d groups test data:\n---------------------------------------\n", _test_groupCount);
		printResult(OutputNodeCount, _test_groupCount, test_output, _test_expectData);
		delete test_output;
	}
}

void NeuralNet::printResult(int nodeCount, int groupCount, Matrix* output, Matrix* expect)
{
	if (_option.getInt("ForceOutput") || groupCount <= 100)
	{
		for (int i = 0; i < groupCount; i++)
		{
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4lf ", output->getData(j, i));
			}
			fprintf(stdout, " --> ");
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4lf ", expect->getData(j, i));
			}
			fprintf(stdout, "\n");
		}
	}

	if (_option.getInt("TestMax"))
	{
		auto outputMax = new Matrix(nodeCount, groupCount, 1, 0);
		outputMax->initData(0);
		auto om = new int[groupCount];
		auto em = new int[groupCount];

		//�Ǳ߱����ֵ�군�ˣ��ֶ��ر�
		for (int i = 0; i < groupCount; i++)
		{
			outputMax->getData(output->indexColMaxAbs(i), i) = 1;
		}

		for (int i = 0; i < groupCount; i++)
		{
			for (int j = 0; j < nodeCount; j++)
			{
				if (outputMax->getData(j, i) == 1)
					om[i] = j;
			}
			for (int j = 0; j < nodeCount; j++)
			{
				if (expect->getData(j, i) == 1)
					em[i] = j;
			}
		}
		if (_option.getInt("ForceOutput") || groupCount <= 100)
		{
			for (int i = 0; i < groupCount; i++)
			{
				//if (om[i] != em[i])
				{
					int om = 0, em = 0;
					for (int j = 0; j < nodeCount; j++)
					{
						if (outputMax->getData(j, i) == 1)
							fprintf(stdout, "%3d (%6.4lf) ", j, output->getData(j, i));
					}
					fprintf(stdout, " --> ");
					for (int j = 0; j < nodeCount; j++)
					{
						if (expect->getData(j, i) == 1)
							fprintf(stdout, "%3d ", j);
					}
					fprintf(stdout, "\n");
				}
			}
		}

		double n = 0;
		for (int i = 0; i < nodeCount*groupCount; i++)
			n += std::abs(outputMax->getData(i) - expect->getData(i));
		n /= 2;
		delete outputMax;
		delete om;
		delete em;
		fprintf(stdout, "Error of max value position: %d, %5.2lf%%\n", int(n), n / groupCount * 100);
	}
}


