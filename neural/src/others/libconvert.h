#pragma once
#define _USE_MATH_DEFINES 
#include <string>
#include <vector>
#include <stdarg.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//string functions
std::string readStringFromFile(const std::string &filename);
void writeStringToFile(const std::string &str, const std::string &filename);
int replaceString(std::string &s, const std::string &oldstring, const std::string &newstring, int pos0 = 0);
int replaceAllString(std::string &s, const std::string &oldstring, const std::string &newstring);
void replaceStringInFile(const std::string &oldfilename, const std::string &newfilename, const std::string &oldstring, const std::string &newstring);
void replaceAllStringInFile(const std::string &oldfilename, const std::string &newfilename, const std::string &oldstring, const std::string &newstring);
std::string formatString(const char *format, ...);
void formatAppendString(std::string &str, const char *format, ...);
std::string findANumber(const std::string &s);
unsigned findTheLast(const std::string &s, const std::string &content);
std::vector<std::string> splitString(std::string str, std::string pattern);
bool isProChar(char c);

template<typename T>
int findNumbers(const std::string &s, std::vector<T> *data)
{
    int n = 0;
    std::string str = "";
    bool haveNum = false;
    for (int i = 0; i < s.length(); i++)
    {
        char c = s[i];
        bool findNumChar = (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'E' || c == 'e';
        if (findNumChar)
        {
            str += c;
            if (c >= '0' && c <= '9')
                haveNum = true;
        }
        if (!findNumChar || i == s.length() - 1)
        {
            if (str != "" && haveNum)
            {
                auto f = T(atof(str.c_str()));
                data->push_back(f);
                n++;
            }
            str = "";
            haveNum = false;
        }
    }
    return n;
}
