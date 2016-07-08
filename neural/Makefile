CXX=g++
CPPFLAGS=-I. -I./lib -O3 -std=c++11 $(EXCPPFLAGS)
LDFLAGS=-L. -lcudart -lcublas -lcudnn $(EXLDFLAGS)
SOURCES=$(wildcard *.cpp) $(wildcard lib/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET=neural

.PHONY: all clean

all: .d $(SOURCES) $(TARGET)

.d: $(SOURCES)
	$(CXX) $(CPPFLAGS) -MM $(SOURCES) >.d
-include .d
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm *.o lib/*.o

gcc: EXCPPFLAGS=-floop-parallelize-all -ftree-vectorize -flto
gcc: EXLDFLAGS=-lpthread -flto -fwhole-program -lopenblas
gcc: all

icc: CXX=icpc
icc: EXCPPFLAGS=-xHost -ipo -finline-functions -no-inline-factor
icc: LDFLAGS=-ipo -mkl
icc: all

clang: CXX=clang++
clang: EXCPPFLAGS=-lopenblas
clang: EXLDFLAGS=
clang: all
