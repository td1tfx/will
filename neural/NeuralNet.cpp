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
	if (_train_inputData) 
		delete _train_inputData;
	if (_train_expectData)	
		delete _train_expectData;
	if (_test_inputData) 
		delete _test_inputData;
	if (_test_expectData) 
		delete _test_expectData;
}

//运行，注意容错保护较弱
void NeuralNet::run()
{
	BatchMode = NeuralNetBatchMode(_option.BatchMode);
	MiniBatchCount = std::max(1, _option.MiniBatch);
	WorkMode = NeuralNetWorkMode(_option.WorkMode);

	LearnSpeed = _option.LearnSpeed;
	Lambda = _option.Regular;

	MaxGroup = _option.MaxGroup;

	if (_option.UseCUDA)
	{
		d_matrix::initCublas();
	}
	if (_option.UseMNIST == 0)
	{
		if (_option.TrainDataFile != "")
		{
			readData(_option.TrainDataFile.c_str(), Train);
		}
	}
	else
	{
		readMNIST();
	}

	//读不到文件强制重新创建网络，不太正常
	//if (readStringFromFile(_option.LoadFile) == "")
	//	_option.LoadNet == 0;

	std::vector<double> v;
	int n = findNumbers(_option.NodePerLayer, v);

	if (_option.LoadNet == 0)
		createByData(_option.Layer, int(v[0]));
	else
		createByLoad(_option.LoadFile.c_str());

	//net->selectTest();
	train(int(_option.TrainTimes), int(_option.OutputInterval), _option.Tol, _option.Dtol);
	test();

	if (_option.SaveFile != "")
		saveInfo(_option.SaveFile.c_str());
	if (_option.TestDataFile != "")
	{
		readData(_option.TestDataFile.c_str(), Test);
		test();
	}
}

//设置学习模式
void NeuralNet::setLearnMode(NeuralNetBatchMode lm, int lb /*= -1*/)
{
	BatchMode = lm;
	//批量学习时，节点数据量等于实际数据量
	if (BatchMode == Online)
	{
		MiniBatchCount = 1;
	}
	//这里最好是能整除的
	if (BatchMode == MiniBatch)
	{
		MiniBatchCount = lb;
	}
}

void NeuralNet::setWorkMode(NeuralNetWorkMode wm)
{
	 WorkMode = wm; 
	 if (wm == Probability)
	 {
		 getLastLayer()->setActiveFunction(Softmax);
	 }
	 if (wm == Classify)
	 {
		 getLastLayer()->setActiveFunction(Findmax);
	 }
}

//创建神经层
void NeuralNet::createLayers(int layerCount)
{
	Layers.resize(layerCount);
	for (int i = 0; i < layerCount; i++)
	{
		auto layer = NeuralLayerFactory::createLayer(FullConnection);
		layer->Id = i;
		Layers[i] = layer;
	}
}


//不能整除的情况未处理
void NeuralNet::active(d_matrix* input, d_matrix* expect, d_matrix* output, int groupCount, int batchCount,
	bool learn /*= false*/, double* error /*= nullptr*/)
{
	if (error) *error = 0;
	for (int i = 0; i < groupCount; i += batchCount)
	{
		int n = resetGroupCount(std::min(batchCount, groupCount-i));
		if (input)
		{
			setInputData(input, i);
		}
		if (expect)
		{
			setExpectData(expect, i);
		}

		for (int i_layer = 1; i_layer < getLayerCount(); i_layer++)
		{
			Layers[i_layer]->activeOutputValue();
		}

		if (learn)
		{
			for (int i_layer = getLayerCount() - 1; i_layer > 0; i_layer--)
			{
				Layers[i_layer]->backPropagate(LearnSpeed, Lambda);
			}
		}
		if (output)
		{
			getOutputData(output, n, i);
		}
		//计算误差，注意这个算法对于minibatch不严格
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

void NeuralNet::setInputData(d_matrix* input, int groupid)
{
	getFirstLayer()->getOutputMatrix()->resetDataPointer(input->getDataPointer(0, groupid));
}

void NeuralNet::setExpectData(d_matrix* expect, int groupid)
{
	getLastLayer()->getExpectMatrix()->resetDataPointer(expect->getDataPointer(0, groupid));
}


void NeuralNet::getOutputData(d_matrix* output, int groupCount, int col/*=0*/)
{
	getLastLayer()->getOutputMatrix()->memcpyDataOut(output->getDataPointer(0, col), OutputNodeCount*groupCount);
}


//训练一批数据，输出步数和误差，若训练次数为0可以理解为纯测试模式
void NeuralNet::train(int times /*= 1000000*/, int interval /*= 1000*/, double tol /*= 1e-3*/, double dtol /*= 1e-9*/)
{
	if (times <= 0) return;
	//这里计算初始的误差，如果足够小就不训练了
	//这个误差是总体误差，与批量误差有区别，故有时首次训练会出现误差增加
	double e = 0;
	active(_train_inputData, _train_expectData, nullptr, _train_groupCount, MiniBatchCount, false, &e);	
	fprintf(stdout, "step = %e, mse = %e\n", 0.0, e);
	if (e < tol) return;
	double e0 = e;
	
	switch (BatchMode)
	{
	case Batch:
		MiniBatchCount = resetGroupCount(_train_groupCount);
		break;
	case Online:
		resetGroupCount(1);
		MiniBatchCount = 1;
		break;
	case MiniBatch:
		if (MiniBatchCount > 0)
			resetGroupCount(MiniBatchCount);
		break;
	default:
		break;
	}

	//训练过程
	e = 0;
	for (int count = 1; count <= times; count++)
	{
		//getFirstLayer()->step = count;
		active(_train_inputData, _train_expectData, nullptr, _train_groupCount, MiniBatchCount, true, count % interval == 0?&e:nullptr);
		if (count % interval == 0)
		{
			fprintf(stdout, "step = %e, mse = %e, diff(mse) = %e\n", double(count), e, e0 - e);
			if (e < tol || std::abs(e - e0) < dtol) break;
			e0 = e;
			e = 0;
		}
	}
}

//读取数据
//这里的处理可能不是很好
void NeuralNet::readData(const char* filename, DateMode dm/*= Train*/)
{
	_train_groupCount = 0;
	_test_groupCount = 0;

	int mark = 3;
	//数据格式：前两个是输入变量数和输出变量数，之后依次是每组的输入和输出，是否有回车不重要
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
	if (dm == Test)
	{
		groupCount = &_test_groupCount;
		inputData = &_test_inputData;
		expectData = &_test_expectData;
	}

	*groupCount = (n - mark) / (InputNodeCount + OutputNodeCount);
	*inputData = new d_matrix(InputNodeCount, *groupCount, 1, 0);
	*expectData = new d_matrix(OutputNodeCount, *groupCount, 1, 0);

	//写法太难看了
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
	for (auto l : Layers)
	{
		l->resetData(n);
	}
	return n;
}

//依据输入数据创建神经网，网络的节点数只对隐藏层有用
//此处是具体的网络结构
void NeuralNet::createByData(int layerCount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	this->createLayers(layerCount);

	getFirstLayer()->initData(InputNodeCount, MiniBatchCount, Input);
	fprintf(stdout, "Layer %d has %d nodes.\n", 0, InputNodeCount);
	for (int i = 1; i < layerCount - 1; i++)
	{
		getLayer(i)->initData(nodesPerLayer, MiniBatchCount, Hidden);
		fprintf(stdout, "Layer %d has %d nodes.\n", i, nodesPerLayer);
	}
	getLastLayer()->initData(OutputNodeCount, MiniBatchCount, Output);
	fprintf(stdout, "Layer %d has %d nodes.\n", layerCount - 1, OutputNodeCount);

	for (int i = 1; i < layerCount; i++)
	{
		Layers[i]->connetPrevlayer(Layers[i - 1]);
	}
}

//输出键结值
void NeuralNet::saveInfo(const char* filename)
{
	FILE *fout = stdout;
	if (filename)
		fout = fopen(filename, "w+t");

	fprintf(fout, "Net information:\n");
	fprintf(fout, "%d\tlayers\n", Layers.size());
	for (int i_layer = 0; i_layer < getLayerCount(); i_layer++)
	{
		fprintf(fout, "layer %d has %d nodes\n", i_layer, Layers[i_layer]->OutputCount);
	}

	fprintf(fout, "---------------------------------------\n");
	for (int i_layer = 1; i_layer < getLayerCount(); i_layer++)
	{
		Layers[i_layer]->saveInfo(fout);
	}

	if (filename)
		fclose(fout);
}

//依据键结值创建神经网
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
	getFirstLayer()->Type = Input;
	getLastLayer()->Type = Output;
	k++;
	for (int i_layer = 0; i_layer < layerCount; i_layer++)
	{
		getLayer(i_layer)->initData(int(v[k]), MiniBatchCount, getLayer(i_layer)->Type);
		fprintf(stdout, "Layer %d has %d nodes.\n", i_layer, int(v[k]));
		k += 2;
	}
	k = 1 + layerCount * 2;
	for (int i_layer = 0; i_layer < layerCount - 1; i_layer++)
	{
		auto& layer1 = Layers[i_layer];
		auto& layer2 = Layers[i_layer + 1];
		layer2->connetPrevlayer(layer1);
		int readcount = layer2->readInfo(v + k, n - k);
		k += readcount;
	}
}

void NeuralNet::readMNIST()
{
	InputNodeCount = 784;
	OutputNodeCount = 10;

	_train_groupCount = 60000;
	_train_inputData = new d_matrix(InputNodeCount, _train_groupCount, 1, 0);
	_train_expectData = new d_matrix(OutputNodeCount, _train_groupCount, 1, 0);

	_test_groupCount = 1000;
	_test_inputData = new d_matrix(InputNodeCount, _train_groupCount, 1, 0);
	_test_expectData = new d_matrix(OutputNodeCount, _train_groupCount, 1, 0);

	MNISTFunctions::readImageFile("train-images.idx3-ubyte", _train_inputData->getDataPointer());
	MNISTFunctions::readLabelFile("train-labels.idx1-ubyte", _train_expectData->getDataPointer());

	MNISTFunctions::readImageFile("t10k-images.idx3-ubyte", _test_inputData->getDataPointer());
	MNISTFunctions::readLabelFile("t10k-labels.idx1-ubyte", _test_expectData->getDataPointer());
}

void NeuralNet::loadOptoin(const char* filename)
{
	_option.loadIni(filename);
}


//这里拆一部分数据为测试数据，写法有hack性质
void NeuralNet::selectTest()
{
	/*
	//备份原来的数据
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

//输出拟合的结果和测试集的结果
void NeuralNet::test()
{
	if (_train_groupCount > 0)
	{		
		auto train_output = new d_matrix(OutputNodeCount, _train_groupCount, 1, 0);
		active(_train_inputData, nullptr, train_output, _train_groupCount, resetGroupCount(_train_groupCount));
		fprintf(stdout, "\n%d groups train data:\n---------------------------------------\n", _train_groupCount);
		printResult(OutputNodeCount, _train_groupCount, train_output, _train_expectData);
		delete train_output;
	}
	if (_test_groupCount > 0)
	{
		auto test_output = new d_matrix(OutputNodeCount, _test_groupCount, 1, 0);
		active(_test_inputData, nullptr, test_output, _test_groupCount, resetGroupCount(_test_groupCount));
		fprintf(stdout, "\n%d groups test data:\n---------------------------------------\n", _test_groupCount);
		printResult(OutputNodeCount, _test_groupCount, test_output, _test_expectData);
		delete test_output;
	}
}

void NeuralNet::printResult(int nodeCount, int groupCount, d_matrix* output, d_matrix* expect)
{
	if (_option.ForceOutput || groupCount <= 100)
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

	if (_option.TestMax)
	{
		auto outputMax = new d_matrix(nodeCount, groupCount, 1, 0);
		outputMax->initData(0);
		auto om = new int[groupCount];
		auto em = new int[groupCount];

		//那边标最大值完蛋了，手动重标
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
		if (_option.ForceOutput || groupCount <= 100)
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


