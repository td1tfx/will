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
public:
    void loadIni(const std::string& filename)
    {
        _ini = new INIReader(filename);
    }

    int getInt(const std::string& name, int v = 0)
    {
        return int(_ini->GetReal("will", name, v));
    }
    double getReal(const std::string& name, double v = 0.0)
    {
        return _ini->GetReal("will", name, v);
    }
    std::string getString(const std::string& name, std::string v = "")
    {
        return _ini->Get("will", name, v);
    }

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


