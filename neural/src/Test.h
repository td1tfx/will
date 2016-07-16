#pragma once
#include "libconvert.h"
#include "types.h"

class Test
{
private:
    static unsigned char* readFile(const char* filename);
    static void reverse(unsigned char* c, int n);
public:
    static int MNIST_readImageFile(const char* filename, real* Xdata);
    static int MNIST_readLabelFile(const char* filename, real* Ydata);


    static void testActive(int tests);
    static void testConvolution(int testc);
    static void testPooling(int testp);

    static void test();
    static void test2();
};

