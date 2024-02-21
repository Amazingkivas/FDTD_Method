#pragma once

#include <fstream>

#include "FDTD.h"

void write_x(Field& this_field, std::ofstream& fout)
{
    for (int k = 0; k < this_field.get_Nk(); ++k)
    {
        for (int j = 0; j < this_field.get_Nj(); ++j)
        {
            for (int i = 0; i < this_field.get_Ni(); ++i)
            {
                fout << this_field(i, j, k);
                if (i == this_field.get_Ni() - 1)
                {
                    fout << std::endl;
                }
                else
                {
                    fout << ";";
                }
            }
        }
    }
    fout << std::endl << std::endl;
}
void write_y(Field& this_field, std::ofstream& fout)
{
    for (int k = 0; k < this_field.get_Nk(); ++k)
    {
        for (int i = 0; i < this_field.get_Ni(); ++i)
        {
            for (int j = 0; j < this_field.get_Nj(); ++j)
            {
                fout << this_field(i, j, k);
                if (j == this_field.get_Nj() - 1)
                {
                    fout << std::endl;
                }
                else
                {
                    fout << ";";
                }
            }
        }
    }
    fout << std::endl << std::endl;
}
void write_z(Field& this_field, std::ofstream& fout)
{
    for (int i = 0; i < this_field.get_Ni(); ++i)
    {
        for (int j = 0; j < this_field.get_Nj(); ++j)
        {
            for (int k = 0; k < this_field.get_Nk(); ++k)
            {
                fout << this_field(i, j, k);
                if (k == this_field.get_Nk() - 1)
                {
                    fout << std::endl;
                }
                else
                {
                    fout << ";";
                }
            }
        }
    }
    fout << std::endl << std::endl;
}

void write_all(FDTD& test, Axis axis, char* file_path)
{
    std::ofstream test_fout;
    test_fout.open(file_path);

    if (!test_fout.is_open())
    {
        throw std::runtime_error("ERROR: Failed to open " + static_cast<std::string>(file_path));
    }
    for (int i = static_cast<int>(Component::EX); i <= static_cast<int>(Component::BZ); ++i)
    {
        if (axis == Axis::X)
            write_x(test.get_field(static_cast<Component>(i)), test_fout);
        else if (axis == Axis::Y)
            write_y(test.get_field(static_cast<Component>(i)), test_fout);
        else 
            write_z(test.get_field(static_cast<Component>(i)), test_fout);
    }
    test_fout.close();
}
