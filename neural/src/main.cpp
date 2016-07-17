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

    auto useCuda = op.getInt("UseCuda");

    Matrix::init(useCuda);
    t.start();
    net.run(&op);
    t.stop();
    Matrix::destroy();

    Test::test2();

    fprintf(stdout, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());

#ifdef _WIN32
    fprintf(stderr, "\nPress any key to exit.\n");
    getchar();
#endif
    return 0;
}
