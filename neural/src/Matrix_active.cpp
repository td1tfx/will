#pragma once
#include "Matrix.h"

void Matrix::activeForward(ActiveFunctionType af, Matrix* X, Matrix* A)
{
	auto nan = CUDNN_NOT_PROPAGATE_NAN;
	switch (af)
	{
	case af_Sigmoid:
		if (X->UseCuda == mc_UseCuda)
		{
			CUDNN_CREATE_SET_DESCIPTOR(X->ActivationDesc,
				cudnnSetActivationDescriptor(X->ActivationDesc, CUDNN_ACTIVATION_SIGMOID, nan, 1));
			cudnnActivationForward(cudnnHandle, X->ActivationDesc, &real_1, X->TensorDesc, X->data, &real_0, A->TensorDesc, A->data);
		}
		else
		{
			VectorMath::sigmoid_v(X->data, A->data, A->max_script);
		}
		break;
	case af_ReLU:
		if (X->UseCuda == mc_UseCuda)
		{
			CUDNN_CREATE_SET_DESCIPTOR(X->ActivationDesc,
				cudnnSetActivationDescriptor(X->ActivationDesc, CUDNN_ACTIVATION_RELU, nan, 1));
			cudnnActivationForward(cudnnHandle, X->ActivationDesc, &real_1, X->TensorDesc, X->data, &real_0, A->TensorDesc, A->data);
		}
		else
		{
			VectorMath::relu_v(X->data, A->data, A->max_script);
		}
		break;
	case af_Tanh:
		if (X->UseCuda == mc_UseCuda)
		{
			CUDNN_CREATE_SET_DESCIPTOR(X->ActivationDesc,
				cudnnSetActivationDescriptor(X->ActivationDesc, CUDNN_ACTIVATION_TANH, nan, 1));
			cudnnActivationForward(cudnnHandle, X->ActivationDesc, &real_1, X->TensorDesc, X->data, &real_0, A->TensorDesc, A->data);
		}
		else
		{
			VectorMath::tanh_v(X->data, A->data, A->max_script);
		}
		break;
	case af_Softmax:
		if (X->UseCuda == mc_UseCuda)
		{
			CUDNN_CREATE_SET_DESCIPTOR(X->asTensorDesc,
				setTensorDesc(X->asTensorDesc, X->col, 1, 1, X->row));
			cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&real_1, X->asTensorDesc, X->data, &real_0, X->asTensorDesc, A->data);
		}
		else
		{
			//因为数值问题，可能需要减去每列最大值
			Matrix::cpyData(A, X);
			for (int i = 0; i < A->col; i++)
			{
				VectorMath::sub_max(A->getDataPointer(0, i), A->row);
			}
			VectorMath::exp_v(A->data, A->data, A->max_script);
			for (int i = 0; i < A->col; i++)
			{
				real sum = A->sumColAbs(i);
				if (sum == 0) continue;
				A->colMultiply(1 / sum, i);
			}
		}
		break;
	case af_SoftmaxLoss:
		if (X->UseCuda == mc_UseCuda)
		{
			CUDNN_CREATE_SET_DESCIPTOR(X->asTensorDesc,
				setTensorDesc(X->asTensorDesc, A->col, 1, 1, A->row));
			cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
				&real_1, X->asTensorDesc, X->data, &real_0, X->asTensorDesc, A->data);
		}
		else
		{
			activeForward(af_Softmax, X, A);
			VectorMath::log_v(A->data, A->data, A->max_script);
		}
		break;
	case af_Linear:
		cpyData(A, X);
		break;
	case af_Findmax:
		//计算时尽量不要使用，只用在验证时
		if (A->max_script <= 0) return;
		if (X->UseCuda == mc_UseCuda)
		{
			auto T = new Matrix(A->row, A->col, md_Inside, mc_NoCuda);
			T->initData(0);
			for (int i_group = 0; i_group < A->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				T->getData(index, i_group) = 1;
			}
			cpyData(A, T);
			delete T;
		}
		else
		{
			A->initData(0);
			for (int i_group = 0; i_group < A->col; i_group++)
			{
				int index = X->indexColMaxAbs(i_group);
				A->getData(index, i_group) = 1;
			}
		}
		break;
	case af_Softplus:
		//GPU部分不支持
		if (X->UseCuda == mc_UseCuda)
		{
			fprintf(stderr, "Unsupport softplus on GPU!\n");
		}
		else
		{
			VectorMath::softplus_v(X->data, A->data, A->max_script);
		}
		break;
	default:
		fprintf(stderr, "Parameters not enough!\n");
		break;
	}
}

void Matrix::activeBackward(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX)
{
	auto nan = CUDNN_NOT_PROPAGATE_NAN;
	switch (af)
	{
	case af_Sigmoid:
		if (X->UseCuda == mc_UseCuda)
		{
			if (X->ActivationDesc)
				cudnnActivationBackward(cudnnHandle, X->ActivationDesc, &real_1, A->TensorDesc, A->data, dA->TensorDesc, dA->data,
					X->TensorDesc, X->data, &real_0, dX->TensorDesc, dX->data);
		}
		else
		{
			VectorMath::sigmoid_vb(A->data, dA->data, X->data, dX->data, dX->max_script);
		}
		break;
	case af_ReLU:
		if (X->UseCuda == mc_UseCuda)
		{
			if (X->ActivationDesc)
				cudnnActivationBackward(cudnnHandle, X->ActivationDesc, &real_1, A->TensorDesc, A->data, dA->TensorDesc, dA->data,
					X->TensorDesc, X->data, &real_0, dX->TensorDesc, dX->data);
		}
		else
		{
			VectorMath::relu_vb(A->data, dA->data, X->data, dX->data, dX->max_script);
		}
		break;
	case af_Tanh:
		//两者结果在1e-10的精度有区别
		if (X->UseCuda == mc_UseCuda)
		{
			if (X->ActivationDesc)
				cudnnActivationBackward(cudnnHandle, X->ActivationDesc, &real_1, A->TensorDesc, A->data, dA->TensorDesc, dA->data,
					X->TensorDesc, X->data, &real_0, dX->TensorDesc, dX->data);
		}
		else
		{
			VectorMath::tanh_vb(A->data, dA->data, X->data, dX->data, dX->max_script);
		}
		break;
	case af_Softmax:
		if (X->UseCuda == mc_UseCuda)
		{
			auto s = cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&real_1, X->asTensorDesc, A->data, X->asTensorDesc, dA->data, &real_0, X->asTensorDesc, dX->data);
		}
		else
		{
			for (int i = 0; i < dX->col; i++)
			{
				auto v = dot(A, i, dA, i);
				VectorMath::softmax_vb_sub(A->getDataPointer(0, i), dA->getDataPointer(0, i), v, dX->getDataPointer(0, i), dX->row);
			}
		}
		break;
	case af_SoftmaxLoss:
		if (X->UseCuda == mc_UseCuda)
		{
			auto s = cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
				&real_1, X->asTensorDesc, A->data, X->asTensorDesc, dA->data, &real_0, X->asTensorDesc, dX->data);
		}
		else
		{
			for (int i = 0; i < dX->col; i++)
			{
				real v = 0;
				for (int j = 0; j < dX->row; j++)
				{
					v += dA->getData(i, j);
				}
				VectorMath::softmaxloss_vb_sub(A->getDataPointer(0, i), dA->getDataPointer(0, i), v, dX->getDataPointer(0, i), dX->row);
			}
		}
		break;
	case af_Linear:
		dX->initData(1);
		break;
	case af_Findmax:
		//似乎应该是返回一个常数矩阵，若考虑效率应当留空此处在外部处理
		dX->initData(1);
		break;
	case af_Softplus:
		//该函数导数就是sigmoid
		activeForward(af_Sigmoid, X, dX);
		break;
	default:
		fprintf(stderr, "Parameters not enough!\n");
		break;
	}
}

//参数更多的的激活函数，包含了前面的功能，如不考虑效率只用这个也可以
//调用时请自己保证参数数量的正确性！
void Matrix::activeForwardEx(ActiveFunctionType af, Matrix* X, Matrix* A,
	std::initializer_list<real> r_list, std::initializer_list<int> i_list, std::initializer_list<Matrix*> as_list)
{
	auto nan = CUDNN_NOT_PROPAGATE_NAN;
	std::vector<real> vr = r_list;
	std::vector<int> vi = i_list;
	std::vector<Matrix*> as = as_list;
	switch (af)
	{
	case af_ClippedReLU:
		if (X->UseCuda == mc_UseCuda)
		{
			CUDNN_CREATE_SET_DESCIPTOR(X->ActivationDesc,
				cudnnSetActivationDescriptor(X->ActivationDesc, CUDNN_ACTIVATION_CLIPPED_RELU, nan, vr[0]));
			cudnnActivationForward(cudnnHandle, X->ActivationDesc, &real_1, X->TensorDesc, X->data, &real_0, A->TensorDesc, A->data);
		}
		else
		{
			VectorMath::clipped_relu_v(X->data, A->data, vr[0], A->max_script);
		}
		break;
	case af_Dropout:
		dropoutForward(X, A, as[1], as[2], vr[0], vi[0]);
		break;
	case af_DivisiveNormalization:
		divisiveNormalizationForward(X, A, as[0], as[1], as[2], vi[0], vr[0], vr[1], vr[2]);
		break;
	case af_BatchNormalization:
		if (X->UseCuda == mc_UseCuda)
		{
			auto s = cudnnBatchNormalizationForwardInference(cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION, &real_1, &real_0,
				X->TensorDesc, X->data, A->TensorDesc, A->data,
				as[1]->TensorDesc, as[1]->data, as[1]->data, as[1]->data, as[1]->data, vr[0]);
			fprintf(stderr, "BatchNormalization status %d\n", s);
		}
		else
		{

		}
		break;
	default:
		activeForward(af, X, A);
		break;
	}
}

void Matrix::activeBackwardEx(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
	std::initializer_list<real> r_list, std::initializer_list<int> i_list, std::initializer_list<Matrix*> as_list)
{
	auto nan = CUDNN_NOT_PROPAGATE_NAN;
	std::vector<real> vr = r_list;
	std::vector<int> vi = i_list;
	std::vector<Matrix*> as = as_list;
	switch (af)
	{
	case af_ClippedReLU:
		if (X->UseCuda == mc_UseCuda)
		{
			if (X->ActivationDesc)
				cudnnActivationBackward(cudnnHandle, X->ActivationDesc, &real_1, A->TensorDesc, A->data, dA->TensorDesc, dA->data,
					X->TensorDesc, X->data, &real_0, dX->TensorDesc, dX->data);
		}
		else
		{
			VectorMath::clipped_relu_vb(A->data, dA->data, X->data, dX->data, vr[0], dX->max_script);
		}
		break;
	case af_Dropout:
		dropoutBackward(A, dA, X, dX, as[1], as[2], vr[0]);
		break;
	case af_DivisiveNormalization:
		divisiveNormalizationBackward(A, dA, X, dX, as[0], as[1], as[2], as[3]);
		break;
	case af_BatchNormalization:
		if (X->UseCuda == mc_UseCuda)
		{
			cudnnBatchNormalizationBackward(cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION, &real_1, &real_0, &real_1, &real_0, X->TensorDesc,
				X->data, dA->TensorDesc, dA->data, dX->TensorDesc, dX->data,
				as[1]->TensorDesc, as[1]->data, as[1]->data, as[1]->data, vr[0], as[1]->data, as[1]->data);
		}
		else
		{

		}
		break;
	default:
		activeBackward(af, A, dA, X, dX);
		break;
	}
}