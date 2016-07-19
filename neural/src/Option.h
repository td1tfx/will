#pragma once
#include <string>
#include "INIReader.h"

//注意实数只获取双精度数，如果是单精度模式会包含隐式转换
//获取整数的时候，先获取双精度数并强制转换
struct Option
{
    Option() {}
    Option(const std::string& filename) { loadIni(filename); }
    ~Option() { delete _ini; }

private:
    INIReader* _ini;
    std::string _default_section = "will";
public:
    void loadIni(const std::string& filename)
    {
        _ini = new INIReader(filename);
    }
    //默认section
    void setDefautlSection(const std::string& section)
    {
        _default_section = section;
    }
    //从默认section提取
    int getInt(const std::string& name, int v = 0)
    {
        return getReal(_default_section, name, v);
    }
    double getReal(const std::string& name, double v = 0.0)
    {
        return getReal(_default_section, name, v);
    }
    std::string getString(const std::string& name, std::string v = "")
    {
        return getString(_default_section, name, v);
    }
    //从指定sectoin提取
    int getInt(const std::string& section, const std::string& name, int v = 0)
    {
        return int(_ini->GetReal(section, name, v));
    }
    double getReal(const std::string& section, const std::string& name, double v = 0.0)
    {
        return _ini->GetReal(section, name, v);
    }
    std::string getString(const std::string& section, const std::string& name, std::string v = "")
    {
        return _ini->Get(section, name, v);
    }

};


