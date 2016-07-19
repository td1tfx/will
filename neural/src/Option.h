#pragma once
#include <string>
#include "INIReader.h"

//ע��ʵ��ֻ��ȡ˫������������ǵ�����ģʽ�������ʽת��
//��ȡ������ʱ���Ȼ�ȡ˫��������ǿ��ת��
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
    //Ĭ��section
    void setDefautlSection(const std::string& section)
    {
        _default_section = section;
    }
    //��Ĭ��section��ȡ
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
    //��ָ��sectoin��ȡ
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


