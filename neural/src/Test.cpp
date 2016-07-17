#include "Test.h"
#include "Matrix.h"
#include "Layer.h"

unsigned char* Test::readFile(const char* filename)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Can not open file %s\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int length = ftell(fp);
    fseek(fp, 0, 0);
    auto s = new unsigned char[length + 1];
    for (int i = 0; i <= length; s[i++] = '\0');
    fread(s, length, 1, fp);
    fclose(fp);
    return s;
}

void Test::reverse(unsigned char* c, int n)
{
    for (int i = 0; i < n / 2; i++)
    {
        auto& a = *(c + i);
        auto& b = *(c + n - 1 - i);
        auto t = b;
        b = a;
        a = t;
    }
}

int Test::MNIST_readImageFile(const char* filename, real* Xdata)
{
    auto content = readFile(filename);
    reverse(content + 4, 4);
    reverse(content + 8, 4);
    reverse(content + 12, 4);
    int count = *(int*)(content + 4);
    int w = *(int*)(content + 8);
    int h = *(int*)(content + 12);
    //fprintf(stderr, "%-30s%d groups data, w = %d, h = %d\n", filename, count, w, h);
    int size = count * w * h;
    //input = new double[size];
    memset(Xdata, 0, sizeof(real)*size);
    for (int i = 0; i < size; i++)
    {
        auto v = *(content + 16 + i);
        Xdata[i] = v / 255.0;
    }

    //  int check = 59990;
    //      for (int i = 784 * check; i < 784*(check+10); i++)
    //  {
    //      if (input[i] != 0)
    //          fprintf(stdout,"1", input[i]);
    //      else
    //          fprintf(stdout," ");
    //      if (i % 28 == 27)
    //          fprintf(stdout,"\n");
    //  }

    delete[] content;
    return w * h;
}

int Test::MNIST_readLabelFile(const char* filename, real* Ydata)
{
    auto content = readFile(filename);
    reverse(content + 4, 4);
    int count = *(int*)(content + 4);
    //fprintf(stderr, "%-30s%d groups data\n", filename, count);
    //expect = new double[count*10];
    memset(Ydata, 0, sizeof(real)*count * 10);
    for (int i = 0; i < count; i++)
    {
        int pos = *(content + 8 + i);
        Ydata[i * 10 + pos] = 1;
    }
    delete[] content;
    return 10;
}


void Test::testActive(int tests)
{
    if (tests)
    {
        Matrix X(4, 4, 1, 1), A(4, 4, 1, 1);
        Matrix dX(4, 4, 1, 1), dA(4, 4, 1, 1);
        Matrix as1(4, 4, 1, 1), as2(4, 4, 1, 1), as3(4, 4, 1, 1), as4(4, 4, 1, 1);

        dA.initData(1);
        X.initRandom();
        real v = 0.5;
        as1.initData(1);
        Matrix::activeForwardEx(af_Dropout, &X, &A, { 0.5 }, { 0 }, { &as1, &as2, &as3, &as4 });
        fprintf(stdout, "X:\n");
        X.print();
        fprintf(stdout, "A:\n");
        A.print();
        Matrix::activeBackwardEx(af_Dropout, &A, &dA, &X, &dX, { 0.5 }, { 9 }, { &as1, &as2, &as3, &as4 });
        fprintf(stdout, "dA:\n");
        dA.print();
        fprintf(stdout, "dX:\n");
        dX.print();
    }
}

void Test::testConvolution(int testc)
{
    if (testc)
    {
        fprintf(stdout, "\nconvolution test:\n");
        int c = 1;
        int n = 2;
        int kc = 1;
        Matrix X(4, 4, 1, n);
        Matrix dX(4, 4, 1, n);

        Matrix W(2, 2, 1, 1);
        Matrix dW(2, 2, 1, 1);

        Matrix A(3, 3, 1, n);
        Matrix dA(3, 3, 1, n);
        Matrix dB(1, 1, 1, 1);

        A.initData(0);
        dX.initData(12);
        X.initData(1, 1);
        W.initData(1);
        dA.initData(1);
        Matrix::convolutionForward(&X, &W, &A);

        fprintf(stdout, "X\n");
        X.print();
        fprintf(stdout, "W\n");
        W.print();
        fprintf(stdout, "A\n");
        A.print();
        fprintf(stdout, "\n");
        fprintf(stdout, "dA\n");
        dA.print();

        Matrix::convolutionBackward(&A, &dA, &X, &dX, &W, &dW, &dB);
        fprintf(stdout, "dX\n");
        dX.print();
        fprintf(stdout, "dW\n");
        dW.print();
        fprintf(stdout, "dB\n");
        dB.print();

        Matrix a(4, 4, md_Outside);
        Matrix r(3, 3, md_Outside);
    }
}

void Test::testPooling(int testp)
{
    if (testp)
    {
        fprintf(stdout, "\npooling test:\n");
        Matrix X(3, 3, 1, 1);
        X.initRandom();
        X.print();
        Matrix A(2, 2, 1, 1);
        auto m = new int[X.getDataCount()];
        auto re = pl_Average_Padding;
        Matrix::poolingForward(re, &X, &A, 2, 2, 2, 2, m);
        fprintf(stdout, "\n");
        A.print();
        for (int i = 0; i < X.getDataCount(); i++)
        {
            //fprintf(stdout,"%d ", m[i]);
        }
        fprintf(stdout, "\n");
        Matrix dX(3, 3, 1, 1);
        Matrix dA(2, 2, 1, 1);
        dA.initData(0, 1);
        Matrix::poolingBackward(re, &A, &dA, &X, &dX, 2, 2, 2, 2, m);
        dA.print();
        fprintf(stdout, "\n");
        dX.print();
        delete m;
    }
}

void Test::test()
{
    testPooling(0);
    testConvolution(0);
    testActive(1);
}

void Test::test2()
{
    Matrix::init(1);
    fprintf(stdout, "Use Cuda\n");
    test();
    Matrix::destroy();
    fprintf(stdout, "No Cuda\n");
    test();
    //printf();
}
