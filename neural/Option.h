#pragma once
#include <string>
#include "lib/INIReader.h"

struct Option
{
#define OPTION_PARAMETERS \
OPTION_STRING(TrainDataFile, "");\
OPTION_STRING(LoadFile, "");\
OPTION_STRING(SaveFile, "");\
OPTION_STRING(TestDataFile, "");\
\
OPTION_INT(UseMNIST, 0);\
OPTION_INT(LoadNet, 0);\
OPTION_INT(TestMax, 0);\
OPTION_INT(ForceOutput, 0);\
\
OPTION_INT(Layer, 3);\
OPTION_STRING(NodePerLayer, "7");\
\
OPTION_INT(LearnMode, 0);\
OPTION_INT(WorkMode, 0);\
OPTION_INT(MiniBatch, -1);\
\
OPTION_DOUBLE(TrainTimes, 100);\
OPTION_DOUBLE(OutputInterval, 1);\
\
OPTION_DOUBLE(LearnSpeed, 0.5);\
OPTION_DOUBLE(Regular, 0.01);\
OPTION_DOUBLE(Tol, 1e-3);\
OPTION_DOUBLE(Dtol, 0);\
\
OPTION_INT(UseCUDA, 0);\
OPTION_INT(MaxGroup, 60000);

/*
	大部分含义很明显
    NodePerLayer是一个字串，定义每层节点数，但是不包含输入和输出层
*/
#define OPTION_STRING(a, b) std::string a = (b)
#define OPTION_INT(a, b) int a = (b)
#define OPTION_DOUBLE(a, b) double a = (b)	
OPTION_PARAMETERS
#undef OPTION_STRING
#undef OPTION_INT
#undef OPTION_DOUBLE

	void loadIni(const char* filename)
	{
		INIReader ini(filename);
#define OPTION_STRING(a, b) a = ini.Get("will", #a, (b))
#define OPTION_INT(a, b) a = ini.GetInteger("will", #a, (b))
#define OPTION_DOUBLE(a, b) a = ini.GetReal("will", #a, (b))
OPTION_PARAMETERS
#undef OPTION_STRING
#undef OPTION_INT
#undef OPTION_DOUBLE
	}
#undef OPTION_PARAMETERS
};

