#include "Net.h"
#include "Matrix.h"
#include "Option.h"
#include "Timer.h"

int main(int argc, char* argv[])
{
    Net net;
    Timer t;
    Option op;

    op.loadIni(argc > 1 ? argv[1] : "p.ini");

    Matrix::init(op.getInt("UseCuda"));
    t.start();
    net.init(&op);
    net.run();
    t.stop();
    Matrix::destroy();
    //Test::test2();

    fprintf(stdout, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());

#ifdef _WIN32
    fprintf(stderr, "\nPress any key to exit.\n");
    getchar();
#endif
    return 0;
}
