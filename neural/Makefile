CXX=g++
CPPFLAGS=-I. -I./lib -O3 -std=c++11 -floop-parallelize-all -ftree-vectorize -flto
LDFLAGS=-L. -lopenblas -lpthread -flto -fwhole-program
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
	rm $(OBJECTS)

icpc: CXX=icpc
icpc: all

clang: CXX=clang++
clang: CPPFLAGS=-I. -I./lib -O3 -std=c++11
clang: LDLAGS=-L. -lopenblas
clang: all
