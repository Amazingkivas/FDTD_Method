#include "Writer.h"

void write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it)
{
    std::ofstream test_fout;
    switch (axis)
    {
    case Axis::X:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int k = 0; k < field.get_Nk(); ++k)
            {
                for (int j = 0; j < field.get_Nj(); ++j)
                {
                    test_fout << field(field.get_Ni() / 2, j, k);
                    if (j == field.get_Nj() - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    case Axis::Y:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int i = 0; i < field.get_Nj(); ++i)
            {
                for (int k = 0; k < field.get_Nk(); ++k)
                {
                    test_fout << field(i, field.get_Nj() / 2, k);
                    if (k == field.get_Nk() - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    case Axis::Z:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int i = 0; i < field.get_Nj(); ++i)
            {
                for (int j = 0; j < field.get_Nk(); ++j)
                {
                    test_fout << field(i, j, field.get_Nk() / 2);
                    if (j == field.get_Nk() - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    default: throw std::logic_error("ERROR: Invalid axis");
    }
}
