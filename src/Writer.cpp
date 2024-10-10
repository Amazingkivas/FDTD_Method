#include "Writer.h"

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

void write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it)
{
    std::ofstream test_fout;
    
    for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
    {
        std::cout << "Write : " << it << " -- " << c << std::endl;
        if (axis == Axis::X)
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
        else if (axis == Axis::Y)
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
        else
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
    }
}
