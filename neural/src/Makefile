CXX=g++
CPPFLAGS=-I. -I../include -I../include/cuda -Iothers -O3 -std=c++11 $(EXCPPFLAGS)
LDFLAGS=-L. -lcudart -lcublas -lcudnn $(EXLDFLAGS)
SOURCES=$(wildcard *.cpp) $(wildcard others/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET=neural

.PHONY: show

show: 
	@echo Use one of gcc/icc/clang.

all: .d $(SOURCES) $(TARGET)

.d: $(SOURCES)
	$(CXX) $(CPPFLAGS) -MM $(SOURCES) >.d
-include .d
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm *.o others/*.o

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
