#pragma once
#include <string>

class Neural
{
public:
    Neural();
    ~Neural();

    const std::string DefaultSection = "will";
    void set_console_color(unsigned short color_index);

    template <class T> void safe_delete(T*& pointer)
    {
        if (pointer)
        { delete pointer; }
        pointer = nullptr;
    }
};

