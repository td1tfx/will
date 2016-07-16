#pragma once
#include <cmath>
#include <cstring>

namespace VectorMath
{
#define VECTOR(fv, f) template<typename T> void fv(const T* x, T* a, int size) { for(int i=0;i<size;i++){a[i]=f(x[i]);} }
#define VECTOR_B(fv, content) template<typename T> void fv(const T* a, const T* da,const T* x, T* dx, int size) { for(int i=0;i<size;i++){dx[i]=(content);} }

    template<typename T> inline T sigmoid(T x) { return 1 / (1 + exp(-x)); }
    template<typename T> inline T softplus(T x) { return log(1 + exp(x)); }
    template<typename T> inline T relu(T x) { return x > 0 ? x : 0; }

    VECTOR(log_v, log);
    VECTOR(exp_v, exp);

    VECTOR(sigmoid_v, sigmoid);
    VECTOR(relu_v, relu);
    VECTOR(tanh_v, tanh);
    VECTOR(softplus_v, softplus);
    template<typename T> void linear_v(T* x, T* a, int size) { memcpy(a, x, sizeof(T)*size); }
    template<typename T> void clipped_relu_v(const T* x, T* a, T v, int size)
    {
        for (int i = 0; i < size; i++)
        {
            a[i] = x[i];
            if (a[i] > v) { a[i] = v; }
            else if (a[i] < 0) { a[i] = 0; }
        }
    }

    VECTOR_B(exp_vb, a[i]);
    VECTOR_B(sigmoid_vb, a[i] * (1 - a[i]) * da[i]); //sigmoid导数直接使用a计算
    VECTOR_B(relu_vb, x[i] > 0 ? 1 : 0);
    VECTOR_B(tanh_vb, (1 - a[i] * a[i]) * da[i]);
    VECTOR_B(softplus_vb, sigmoid(x[i]));
    VECTOR_B(linear_vb, 1);

    template<typename T> void clipped_relu_vb(const T* a, const T* da, const T* x, T* dx, T v, int size)
    {
        for (int i = 0; i < size; i++)
        {
            dx[i] = (x[i] > 0) && (x[i] < v) ? 1 : 0;
        }
    }

    //下面3个都是softmax用的
    template<typename T> void sub_max(T* x, int size)
    {
        auto m = x[0];
        for (int i = 1; i < size; i++)
        {
            m = std::max(x[i], m);
        }
        for (int i = 0; i < size; i++)
        {
            x[i] -= m;
        }
    }

    template<typename T> void softmax_vb_sub(const T* a, const T* da, T v, T* dx, int size)
    {
        for (int i = 0; i < size; i++)
        {
            dx[i] = a[i] * (da[i] - v);
        }
    }

    template<typename T> void softmaxloss_vb_sub(const T* a, const T* da, T v, T* dx, int size)
    {
        for (int i = 0; i < size; i++)
        {
            dx[i] = da[i] - v * exp(a[i]);
        }
    }

    template<typename T> bool inbox(T _x, T _y, T x, T y, T w, T h)
    {
        return _x >= x && _y >= y && _x < x + h && _y < y + h;
    }

#undef VECTOR
#undef VECTOR_B
};

