#pragma once
#include "Matrix.h"

void Matrix::setTensorDesc(cudnnTensorDescriptor_t tensor, int n, int c, int h, int w)
{
	if (tensor)
	{
		cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, n, c, h, w);
	}
}

void Matrix::tryInitNullTensorDesc(cudnnTensorDescriptor_t* tensor, int n, int c, int h, int w)
{
	if (*tensor == nullptr)
	{
		cudnnCreateDescriptor(tensor);
		cudnnSetTensor4dDescriptor(*tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, n, c, h, w);
	}
}

//池化，注意利用一个record记录下了对应位置
//gpu部分，平均模式下对padding的支持目前还有问题
void Matrix::poolingForward(ResampleType re, Matrix* X, Matrix* A,
	int window_w, int window_h, int stride_w, int stride_h, int* recordPos /*= nullptr*/)
{
	if (X->UseCuda == mc_UseCuda)
	{
		if (!X->PoolingDesc)
		{
			cudnnCreateDescriptor(&X->PoolingDesc);
			cudnnSetPooling2dDescriptor(X->PoolingDesc, cudnnPoolingMode_t(re), CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);
		}
		cudnnPoolingForward(cudnnHandle, X->PoolingDesc, &real_1, X->TensorDesc, X->data, &real_0, A->TensorDesc, A->data);
	}
	else
	{
		for (int p = 0; p < A->N*A->C; p++)
		{
			for (int wA = 0; wA < A->W; wA++)
			{
				for (int hA = 0; hA < A->H; hA++)
				{
					real v = 0;
					//if (re == re_Average)v = 0;
					if (re == re_Max) v = -REAL_MAX;
					int n = 0;
					for (int wX = wA*stride_w; wX < std::min(X->W, wA*stride_w + window_w); wX++)
					{
						for (int hX = hA*stride_h; hX < std::min(X->H, hA*stride_h + window_h); hX++)
						{
							if (re == re_Average_Padding || re == re_Average_NoPadding)
							{
								v += X->getData(wX, hX, p);
								if (recordPos) recordPos[wX + hX*X->W + p*X->H*X->W] = wA + hA*A->W + p*A->H*A->W;
								n++;
							}
							else if (re == re_Max)
							{
								auto x = X->getData(wX, hX, p);
								if (x > v)
								{
									v = x;
									if (recordPos) recordPos[wA + hA*A->W + p*A->H*A->W] = wX + hX*X->W + p*X->H*X->W;
								}
							}
						}
					}
					if (re == re_Average_Padding)
					{
						v /= window_w*window_h;
					}
					else if (re == re_Average_NoPadding)
					{
						v /= n;
					}
					A->getData(wA, hA, p) = v;
				}
			}
		}
	}
}

//使用cpu时利用了record
void Matrix::poolingBackward(ResampleType re, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
	int window_w, int window_h, int stride_w, int stride_h, int* recordPos /*= nullptr*/)
{
	if (X->UseCuda == mc_UseCuda)
	{
		//这个怎么看都快不了
		if (X->PoolingDesc)
			cudnnPoolingBackward(cudnnHandle, X->PoolingDesc, &real_1, A->TensorDesc, A->data, dA->TensorDesc, dA->data,
				X->TensorDesc, X->data, &real_0, dX->TensorDesc, dX->data);
	}
	else
	{
		if (re == re_Max && recordPos)
		{
			//cpu计算时必须传入一个记录数组，保存最大值的位置，这样速度会快一点
			dX->initData(0);
			for (int i = 0; i < dA->getDataCount(); i++)
			{
				dX->getData(recordPos[i]) = dA->getData(i);
			}
		}
		//对于平均值池化，两种算法实际上遍历元素的数目是相同的
		else if (re == re_Average_Padding && recordPos)
		{
			for (int i = 0; i < dX->getDataCount(); i++)
			{
				dX->getData(i) = dA->getData(recordPos[i]) / window_w / window_h;
			}
		}
		else if ((re == re_Average_Padding && recordPos == nullptr) || re == re_Average_NoPadding)
		{
			for (int p = 0; p < dA->N*dA->C; p++)
			{
				for (int wdA = 0; wdA < dA->W; wdA++)
				{
					for (int hdA = 0; hdA < dA->H; hdA++)
					{
						int n;
						if (re == re_Average_NoPadding)
						{
							n = std::min(window_w, dX->W - wdA*stride_w) * std::min(window_h, dX->H - hdA*stride_h);
						}
						else
						{
							n = window_w * window_h;
						}
						real v = dA->getData(wdA, hdA, p) / n;
						for (int wdX = wdA*stride_w; wdX < std::min(dX->W, wdA*stride_w + window_w); wdX++)
						{
							for (int hdX = hdA*stride_h; hdX < std::min(dX->H, hdA*stride_h + window_h); hdX++)
							{
								dX->getData(wdX, hdX, p) = v;
							}
						}
					}
				}
			}
		}
	}
}

void Matrix::convolutionForward(Matrix* X, Matrix* W, Matrix* A, int* recordX /*= nullptr*/, int* recordW /*= nullptr*/)
{
	if (X->UseCuda == mc_UseCuda)
	{
		cudnnConvolutionFwdAlgoPerf_t cfap[8];
		auto cfa = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
		int n;
		auto scd = CUDNN_STATUS_SUCCESS;
		auto sfd = scd;
		if (!X->ConvolutionDesc)
		{
			cudnnCreateDescriptor(&X->ConvolutionDesc);
			scd = cudnnSetConvolution2dDescriptor(X->ConvolutionDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);
		}
		if (!W->FilterDesc)
		{
			cudnnCreateDescriptor(&W->FilterDesc);
			sfd = cudnnSetFilter4dDescriptor(W->FilterDesc, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, A->C, X->C, W->H, W->W);
		}
		cudnnGetConvolutionForwardAlgorithm(cudnnHandle, X->TensorDesc, W->FilterDesc, X->ConvolutionDesc, A->TensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, workspace_size, &cfa);
		//cudnnFindConvolutionForwardAlgorithm(cudnnHandle, X->tensorDes, fd, cd, A->tensorDes, 8, &n, cfap);

		auto scf = cudnnConvolutionForward(cudnnHandle, &real_1, X->TensorDesc, X->data, W->FilterDesc, W->data, X->ConvolutionDesc,
			cfa, workspace, workspace_size, &real_0, A->TensorDesc, A->data);
		//printf("%d, %d, %d\n", scd, sfd, scf);
	}
	else
	{
		//实际上可以处理为一个大稀疏矩阵乘，太麻烦也不见得会快，不管了
		//除了1CC和CC1，其他的不保证与GPU结果一致
		//if (X->C != 1 && A->C != 1) return;
		A->initData(0);
		convolution_sub(A, W, X, A, W->C, X->N, 1);
	}
}

void Matrix::convolutionBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* W, Matrix* dW, Matrix* dB)
{
	if (X->UseCuda == mc_UseCuda)
	{
		int n;
		cudnnStatus_t scbd, scbf, scbb;
		//cudnnConvolutionBwdDataAlgoPerf_t cbdap[8];
		//cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle, W->FilterDesc, dA->TensorDesc, X->ConvolutionDesc, dX->TensorDesc, 8, &n, cbdap);
		//cudnnConvolutionBwdFilterAlgoPerf_t cbfap[8];
		//cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle, X->TensorDesc, dA->TensorDesc, X->ConvolutionDesc, W->FilterDesc, 8, &n, cbfap);
		if (dX && X->ConvolutionDesc)
		{
			auto cbda = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, W->FilterDesc, dA->TensorDesc, X->ConvolutionDesc, dX->TensorDesc,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, workspace_size, &cbda);
			scbd = cudnnConvolutionBackwardData(cudnnHandle, &real_1, W->FilterDesc, W->data, dA->TensorDesc, dA->data, X->ConvolutionDesc,
				cbda, workspace, workspace_size, &real_0, dX->TensorDesc, dX->data);
		}
		if (dW && X->ConvolutionDesc)
		{
			auto cbfa = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, X->TensorDesc, dA->TensorDesc, X->ConvolutionDesc, W->FilterDesc,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, workspace_size, &cbfa);
			scbf = cudnnConvolutionBackwardFilter(cudnnHandle, &real_1, X->TensorDesc, X->data, dA->TensorDesc, dA->data, X->ConvolutionDesc,
				cbfa, workspace, workspace_size, &real_0, W->FilterDesc, dW->data);
		}
		if (dB)
		{
			scbb = cudnnConvolutionBackwardBias(cudnnHandle, &real_1, dA->TensorDesc, dA->data, &real_0, dB->TensorDesc, dB->data);
		}
		//printf("%d, %d, %d\n", scbd, scbf, scbb);
	}
	else
	{
		if (dX)
		{
			dX->initData(0);
			convolution_sub(dA, W, dX, dX, W->C, dX->N, 1);
		}
		if (dW)
		{
			//N不为1情况下不一致
			dW->initData(0);
			convolution_sub(dA, dW, X, dW, dW->C, X->N, 1);
			//dW->multiply(1.0f*dA->N);
		}
		if (dB)
		{
			dB->initData(0);
			//这个好像就是对对应的A求和
			for (int n = 0; n < dA->N; n++)
			{
				for (int c = 0; c < dA->C; c++)
				{
					for (int h = 0; h < dA->H; h++)
					{
						for (int w = 0; w < dA->W; w++)
						{
							dB->getData(0, 0, c, 0) += dA->getData(w, h, c, n);
						}
					}
				}
			}
		}
	}
}

//R必须是XYZ其中之一！XY循环遍历，，坐标运算X在前，Z判断坐标，plus是加减法
//一般来说应选择维度较小的作为X和Y
//只在CPU运算中起作用
void Matrix::convolution_sub(Matrix* X, Matrix* Y, Matrix* Z, Matrix* R, int C, int N, int plus)
{
	if (R->UseCuda == mc_UseCuda) return;

	for (int n = 0; n < N; n++)
	{
		int nX = n % X->N;
		int nY = n % Y->N;
		int nZ = n % Z->N;
		for (int c = 0; c < C; c++)
		{
			int cX = c % X->C;
			int cY = c % Y->C;
			int cZ = c % Z->C;
			for (int wX = 0; wX < X->W; wX++)
			{
				for (int hX = 0; hX < X->H; hX++)
				{
					for (int wY = 0; wY < Y->W; wY++)
					{
						for (int hY = 0; hY < Y->H; hY++)
						{
							int wZ, hZ;
							if (plus == 1)
							{
								wZ = wX + wY;
								hZ = hX + hY;
							}
							else if (plus == -1)
							{
								wZ = wX - wY;
								hZ = hX - hY;
							}
							if (wZ >= 0 && hZ >= 0 && wZ < Z->W && hZ < Z->H)
							{
								if (R == X)
									X->getData(wX, hX, cX, nX) += Y->getData(wY, hY, cY, nY)*Z->getData(wZ, hZ, cZ, nZ);
								else if (R == Y)
									Y->getData(wY, hY, cY, nY) += X->getData(wX, hX, cX, nX)*Z->getData(wZ, hZ, cZ, nZ);
								else if (R == Z)
									Z->getData(wZ, hZ, cZ, nZ) += X->getData(wX, hX, cX, nX)*Y->getData(wY, hY, cY, nY);
							}
						}
					}
				}
			}
		}
	}
}


void Matrix::dropoutForward(Matrix* X, Matrix* A, Matrix* rgStat, Matrix* stat, real v, int seed/*=0*/)
{
	if (A->UseCuda == mc_UseCuda)
	{
		//会改写as1和as2作为辅助空间
		if (X->DropoutDesc == nullptr)
		{
			size_t size1, size2;
			cudnnDropoutGetStatesSize(cudnnHandle, &size1);
			rgStat->resize(size1 / sizeof(real) + 1, 1);
			cudnnDropoutGetReserveSpaceSize(X->TensorDesc, &size2);
			stat->resize(size2 / sizeof(real) + 1, 1);
			//fprintf(stderr, "dropout size %d,%d\n", size, size2);
			if (!X->DropoutDesc)
			{
				cudnnCreateDescriptor(&X->DropoutDesc);
				cudnnSetDropoutDescriptor(X->DropoutDesc, cudnnHandle, v, rgStat->data, rgStat->getMemerySize(), seed);
			}
		}
		cudnnDropoutForward(cudnnHandle, X->DropoutDesc, X->TensorDesc, X->data, A->TensorDesc, A->data, stat->data, stat->getMemerySize());
	}
	else
	{
		Random<real> r;
		r.reset(seed);
		for (int i = 0; i < A->max_script; i++)
		{
			if (r.rand_uniform() < v)
			{
				A->data[i] = 0;
			}
			else
			{
				A->data[i] = X->data[i] / (1 - v);
			}
		}
	}
}

void Matrix::dropoutBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* rgStat, Matrix* stat, real v)
{
	if (dX->UseCuda == mc_UseCuda)
	{
		if (X->DropoutDesc)
			cudnnDropoutBackward(cudnnHandle, X->DropoutDesc, dA->TensorDesc, dA->data, dX->TensorDesc, dX->data, stat->data, stat->getMemerySize());
	}
	else
	{
		for (int i = 0; i < dX->max_script; i++)
		{
			if (A->data[i] == 0)
			{
				dX->data[i] = 0;
			}
			else
			{
				dX->data[i] = dA->data[i] / (1 - v);
			}
		}
	}
}


void Matrix::divisiveNormalizationForward(Matrix* X, Matrix* A, Matrix* means, Matrix* temp1, Matrix* temp2,
	unsigned lrnN, real lrnAlpha, real lrnBeta, real lrnK)
{
	if (X->UseCuda == mc_UseCuda)
	{
		//先不管了
		if (!X->LRNDesc)
		{
			cudnnCreateDescriptor(&X->LRNDesc);
			cudnnSetLRNDescriptor(X->LRNDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
		}
		cudnnDivisiveNormalizationForward(cudnnHandle, X->LRNDesc, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &real_1,
			X->TensorDesc, X->data, means->data, temp1->data, temp2->data, &real_0, A->TensorDesc, A->data);
	}
	else
	{

	}
}

void Matrix::divisiveNormalizationBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* means, Matrix* temp1, Matrix* temp2, Matrix* dmeans)
{
	if (X->UseCuda == mc_UseCuda)
	{
		if (X->LRNDesc)
			cudnnDivisiveNormalizationBackward(cudnnHandle, X->LRNDesc, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &real_1,
				X->TensorDesc, X->data, means->data, dA->data, temp1->data, temp2->data, &real_0, dX->TensorDesc, dX->data, dmeans->data);
	}
	else
	{

	}
}

void Matrix::batchNormalizationForward(Matrix* X, Matrix* A, Matrix* rgStat, Matrix* stat)
{
	// 	if (X->UseCuda == mc_UseCuda)
	// 	{
	// 		auto s = cudnnBatchNormalizationForwardInference(cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION, &real_1, &real_0,
	// 			X->TensorDesc, X->data, A->TensorDesc, A->data,
	// 			as[1]->TensorDesc, as[1]->data, as[1]->data, as[1]->data, as[1]->data, vr[0]);
	// 		fprintf(stderr, "BatchNormalization status %d\n", s);
	// 	}
	// 	else
	// 	{
	// 
	// 	}

}

void Matrix::batchNormalizationBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* rgStat, Matrix* stat, real v)
{

}


void Matrix::spatialTfSamplerForward()
{

}

void Matrix::spatialTfSamplerBackward()
{

}
