#include "Net.h"
#include "Random.h"

Net::Net()
{

}

Net::~Net()
{
	//safe_delete({ Layers[0],Layers[1],Layers[2] });
	for (int i = 0; i < LayerCount; i++)
	{
		safe_delete(Layers[i]);
		//printf("%d\n", Layers[i]);
	}
	delete[] Layers;
	safe_delete(trainX);
	safe_delete(trainY);
	safe_delete(testX);
	safe_delete(testY);
}

//���У�ע���ݴ�������
void Net::run(Option* op)
{
	BatchType = NetBatchType(op->getInt("BatchMode"));
	MiniBatchCount = std::max(1, op->getInt("MiniBatch"));
	WorkType = NeuralNetWorkType(op->getInt("WorkMode"));

	LearnSpeed = op->getReal("LearnSpeed", 0.5);
	Lambda = op->getReal("Regular");

	MaxGroup = op->getInt("MaxGroup", 100000);
	
	if (op->getInt("UseMNIST") == 0)
	{
		readData(op->getString("TrainDataFile").c_str(), &train_groupCount, &trainX, &trainY);
		readData(op->getString("TestDataFile").c_str(), &test_groupCount, &testX, &testY);
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

	setWorkType(NeuralNetWorkType(op->getInt("WorkType", 0)));

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

//������湤����
Layer* Net::createLayer(LayerConnectionType mode)
{
	Layer* layer = nullptr;

	switch (mode)
	{
	case lc_Full:
		layer = new LayerFull();
		break;
	case lc_Convolution:
		layer = new LayerConvolution();
		break;
	case lc_Pooling:
		layer = new LayerPooling();
		break;
	default:
		break;
	}
	return layer;
}

//����ѧϰģʽ
void Net::setBatchType(NetBatchType bt, int lb /*= -1*/)
{
	BatchType = bt;
	//����ѧϰʱ���ڵ�����������ʵ��������
	if (BatchType == nl_Online)
	{
		MiniBatchCount = 1;
	}
	//�����������������
	if (BatchType == nl_MiniBatch)
	{
		MiniBatchCount = lb;
	}
}

void Net::setWorkType(ActiveFunctionType wt)
{
	WorkType = wt;
	getLastLayer()->setActiveFunction(wt);
}

//�����񾭲�
void Net::createLayers(int layerCount)
{
	Layers = new Layer*[layerCount];
	LayerCount = layerCount;
	for (int i = 0; i < layerCount; i++)
	{
		auto layer = createLayer(lc_Full);
		layer->Id = i;
		Layers[i] = layer;
	}
}


//ѵ��һ�����ݣ��������������ѵ������Ϊ0�������Ϊ������ģʽ
void Net::train(int times, int interval, real tol, real dtol)
{
	if (times <= 0) return;
	//��������ʼ��������㹻С�Ͳ�ѵ����
	//����������������������������𣬹���ʱ�״�ѵ��������������
	real e = 0;
	trainX->tryUploadToCuda();
	trainY->tryUploadToCuda();
	active(trainX, trainY, nullptr, train_groupCount, MiniBatchCount, false, &e);
	fprintf(stdout, "step = %e, mse = %e\n", 0.0, e);
	if (e < tol) return;
	real e0 = e;

	switch (BatchType)
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
		active(trainX, trainY, nullptr, train_groupCount, MiniBatchCount, true, count % interval == 0 ? &e : nullptr);
		if (count % interval == 0 || count == times)
		{
			fprintf(stdout, "step = %e, mse = %e, diff(mse) = %e\n", real(count), e, e0 - e);
			if (e < tol || std::abs(e - e0) < dtol) break;
			e0 = e;
		}
	}
}

void Net::active(Matrix* X, Matrix* Y, Matrix* A, int groupCount, int batchCount,
	bool learn /*= false*/, real* error /*= nullptr*/)
{
	Random<real> r;
	r.set_seed();
	if (error) *error = 0;
	for (int i = 0; i < groupCount; i += batchCount)
	{
		int selectgroup = i;
		if (batchCount <= 1)
		{
			selectgroup = int(r.rand_uniform()*groupCount);
		}
		int n = resetGroupCount(std::min(batchCount, groupCount - selectgroup));
		if (X)
		{
			getFirstLayer()->AMatrix->shareData(X, 0, selectgroup);
		}
		if (Y)
		{
			getLastLayer()->YMatrix->shareData(Y, 0, selectgroup);
		}

		for (int i_layer = 1; i_layer < getLayerCount(); i_layer++)
		{
			Layers[i_layer]->activeForward();
		}

		if (learn)
		{
			for (int i_layer = getLayerCount() - 1; i_layer > 0; i_layer--)
			{
				Layers[i_layer]->activeBackward();
				Layers[i_layer]->updateParameters(LearnSpeed, Lambda);
			}
		}
		if (A)
		{
			getOutputData(A, n, selectgroup);
		}
		//������ע������㷨����minibatch���ϸ�
		if (error)
		{
			if (!learn)
			{
				getLastLayer()->activeBackward();
			}
			*error += getLastLayer()->getdAMatrix()->ddot() / groupCount / OutputNodeCount;
		}
	}
}


void Net::getOutputData(Matrix* M, int groupCount, int col /*= 0*/)
{
	getLastLayer()->getAMatrix()->memcpyDataOutToHost(M->getDataPointer(0, col), OutputNodeCount*groupCount);
}


//��ȡ����
//����Ĵ�����ܲ��Ǻܺ�
void Net::readData(const char* filename, int* count, Matrix** pX, Matrix** pY)
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
	*pX = new Matrix(InputNodeCount, *count, md_Inside, mc_NoCuda);
	*pY = new Matrix(OutputNodeCount, *count, md_Inside, mc_NoCuda);

	//д��̫�ѿ���
	int k = mark, k1 = 0, k2 = 0;

	for (int i_data = 1; i_data <= (*count); i_data++)
	{
		for (int i = 1; i <= InputNodeCount; i++)
		{
			(*pX)->getData(k1++) = v[k++];
		}
		for (int i = 1; i <= OutputNodeCount; i++)
		{
			(*pY)->getData(k2++) = v[k++];
		}
	}
}

int Net::resetGroupCount(int n)
{
	if (n == Layer::GroupCount) return n;
	if (n > MaxGroup)
		n = MaxGroup;
	Layer::setGroupCount(n);
	for (int i = 0; i < LayerCount; i++)
	{
		Layers[i]->resetGroupCount();
	}
	return n;
}

//�����������ݴ�������������Ľڵ���ֻ�����ز�����
//�˴��Ǿ��������ṹ
void Net::createByData(int layerCount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	Layer::setGroupCount(MiniBatchCount);

	this->createLayers(layerCount);

	LayerInitInfo info;

	info.full.outputCount = InputNodeCount;
	getFirstLayer()->initData(lt_Input, &info);
	fprintf(stdout, "Layer %d has %d nodes.\n", 0, InputNodeCount);
	for (int i = 1; i < layerCount - 1; i++)
	{
		info.full.outputCount = nodesPerLayer;
		getLayer(i)->initData(lt_Hidden, &info);
		fprintf(stdout, "Layer %d has %d nodes.\n", i, nodesPerLayer);
	}
	info.full.outputCount = OutputNodeCount;
	getLastLayer()->initData(lt_Output, &info);
	fprintf(stdout, "Layer %d has %d nodes.\n", layerCount - 1, OutputNodeCount);

	for (int i = 1; i < layerCount; i++)
	{
		Layers[i]->connetPrevlayer(Layers[i - 1]);
	}
}

//�������ֵ
void Net::saveInfo(const char* filename)
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
void Net::createByLoad(const char* filename)
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
	LayerInitInfo info;
	for (int i_layer = 0; i_layer < layerCount; i_layer++)
	{
		info.full.outputCount = int(v[k]);
		getLayer(i_layer)->initData(getLayer(i_layer)->Type, &info);
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

void Net::readMNIST()
{
	InputNodeCount = 784;
	OutputNodeCount = 10;

	train_groupCount = 60000;
	trainX = new Matrix(InputNodeCount, train_groupCount, md_Inside, mc_NoCuda);
	trainY = new Matrix(OutputNodeCount, train_groupCount, md_Inside, mc_NoCuda);

	test_groupCount = 10000;
	testX = new Matrix(InputNodeCount, train_groupCount, md_Inside, mc_NoCuda);
	testY = new Matrix(OutputNodeCount, train_groupCount, md_Inside, mc_NoCuda);

	Test::MNIST_readImageFile("train-images.idx3-ubyte", trainX->getDataPointer());
	Test::MNIST_readLabelFile("train-labels.idx1-ubyte", trainY->getDataPointer());

	Test::MNIST_readImageFile("t10k-images.idx3-ubyte", testX->getDataPointer());
	Test::MNIST_readLabelFile("t10k-labels.idx1-ubyte", testY->getDataPointer());
}


void Net::selectTest()
{

}


//�����ϵĽ���Ͳ��Լ��Ľ��
void Net::test(int forceOutput /*= 0*/, int testMax /*= 0*/)
{
	outputTest("train", OutputNodeCount, train_groupCount, trainX, trainY, forceOutput, testMax);
	outputTest("test", OutputNodeCount, test_groupCount, testX, testY, forceOutput, testMax);
}

void Net::extraTest(const char* filename, int forceOutput /*= 0*/, int testMax /*= 0*/)
{
	int count = 0;
	Matrix *X = nullptr, *Y = nullptr;
	readData(filename, &count, &X, &Y);
	outputTest("extra test", OutputNodeCount, count, X, Y, forceOutput, testMax);
	safe_delete(X);
	safe_delete(Y);
}

void Net::outputTest(const char* info, int nodeCount, int groupCount, Matrix* X, Matrix* Y, int forceOutput, int testMax)
{
	if (groupCount <= 0) return;

	Y->tryDownloadFromCuda();
	auto A = new Matrix(nodeCount, groupCount, md_Inside, mc_NoCuda);
	X->tryUploadToCuda();
	active(X, nullptr, A, groupCount, resetGroupCount(groupCount));

	fprintf(stdout, "\n%d groups %s data:\n---------------------------------------\n", groupCount, info);
	if (forceOutput || groupCount <= 100)
	{
		for (int i = 0; i < groupCount; i++)
		{
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4f ", A->getData(j, i));
			}
			fprintf(stdout, " --> ");
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4f ", Y->getData(j, i));
			}
			fprintf(stdout, "\n");
		}
	}
	if (testMax)
	{
		auto AMax = new Matrix(nodeCount, groupCount, md_Inside, mc_NoCuda);
		Matrix::activeForward(af_Findmax, A, AMax);

		if (forceOutput || groupCount <= 100)
		{
			for (int i = 0; i < groupCount; i++)
			{
				int o = AMax->indexColMaxAbs(i);
				int e = Y->indexColMaxAbs(i);
				fprintf(stdout, "%3d (%6.4f) --> %3d\n", o, A->getData(o, i), e);
			}
		}

		real n = 0;
		Matrix::add(AMax, -1, Y, AMax);
		n = AMax->sumAbs() / 2;
		delete AMax;
		fprintf(stdout, "Error of max value position: %d, %5.2f%%\n", int(n), n / groupCount * 100);
	}
	delete A;
}
