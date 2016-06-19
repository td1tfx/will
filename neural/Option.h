#pragma once
#include <string>
#include "lib/INIReader.h"

struct Option
{
	Option() {}
	Option(const char* filename) { loadIni(filename); }
	~Option() { delete _ini; }

private:
	INIReader* _ini;
public:
	void loadIni(const char* filename)
	{
		_ini = new INIReader(filename);
	}

	int getInt(const char* name, int v = 0)
	{
		return int(_ini->GetReal("will", name, v));
	}
	double getDouble(const char* name, double v = 0)
	{
		return _ini->GetReal("will", name, v);
	}
	std::string getString(const char* name, std::string v = "")
	{
		return _ini->Get("will", name, v);
	}

};


