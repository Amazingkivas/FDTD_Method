#define _USE_MATH_DEFINES
//#define __DEBUG__

#include <iostream>
#include <iomanip>
#include <fstream>

#include "FDTD.h"

double sign;
double shift = 0.0;

double get_vector_sign(Component vect_1, Component vect_2)
{
    if (vect_1 == Component::EY && vect_2 == Component::BZ || vect_1 == Component::EZ && vect_2 == Component::BX)
    {
        return 1.0;
    }
    else if (vect_1 == Component::EX && vect_2 == Component::BZ || vect_1 == Component::EZ && vect_2 == Component::BY)
    {
        return -1.0;
    }
    else return 0.0;
}

void initial_filling_x(FDTD& test, Component field_1, Component field_2, int size_N[2], double size_d, double size_wave[2], bool shifted = true)
{
    double x = 0.0;
    double x_b = 0.0;
    for (int i = 0; i < size_N[0]; x += size_d, ++i)
    {
        for (int j = 0; j < size_N[1]; ++j)
        {
            test.get_field(field_1)(i, j) = sign * sin(2.0 * M_PI * (x - size_wave[0]) / (size_wave[1] - size_wave[0]));
            if (shifted)
            {
                x_b = x + size_d / 2.0;
            }
            else x_b = x;
            test.get_field(field_2)(i, j) = sin(2.0 * M_PI * (x_b - size_wave[0]) / (size_wave[1] - size_wave[0]));
        }
    }
}
void initial_filling_y(FDTD& test, Component field_1, Component field_2, int size_N[2], double size_d, double size_wave[2], bool shifted = true)
{
    double y = 0.0;
    double y_b = 0.0;
    for (int j = 0; j < size_N[1]; y += size_d, ++j)
    {
        for (int i = 0; i < size_N[0]; ++i)
        {
            test.get_field(field_1)(i, j) = sign * sin(2.0 * M_PI * (y - size_wave[0]) / (size_wave[1] - size_wave[0]));
            if (shifted)
            {
                y_b = y + size_d / 2.0;
            }
            else y_b = y;
            test.get_field(field_2)(i, j) = sin(2.0 * M_PI * (y_b - size_wave[0]) / (size_wave[1] - size_wave[0]));
        }
    }
}

void write_x(Field& this_field, std::ofstream& fout)
{
    for (int j = 0; j < this_field.get_Nj(); ++j)
    {
        for (int i = 0; i < this_field.get_Ni(); ++i)
        {
            fout << this_field(i, j);
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
    fout << std::endl << std::endl;
}
void write_y(Field& this_field, std::ofstream& fout)
{
    for (int i = 0; i < this_field.get_Ni(); ++i)
    {
        for (int j = 0; j < this_field.get_Nj(); ++j)
        {
            fout << this_field(i, j);
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
    fout << std::endl << std::endl;
}

double max_abs_error_x(Field& this_field, int size_N[2], double size_x[2], double size_d[2], double t, bool shifted = true)
{
    double extr_n = 0.0;
    double x = (shifted) ? shift : 0.0;
    int j = 0;
    for (int i = 0; i < this_field.get_Ni(); ++i, x += size_d[0])
    {
        double this_n = fabs(sign * this_field(i, j) - sin(2.0 * M_PI * (x - size_x[0] - FDTD_Const::C * t) / (size_x[1] - size_x[0])));
        if (this_n > extr_n)
            extr_n = this_n;
    }
    return extr_n;
}
double max_abs_error_y(Field& this_field, int size_N[2], double size_y[2], double size_d[2], double t, bool shifted = true)
{
    double extr_n = 0.0;
    double y = (shifted) ? shift : 0.0;
    int i = 0;
    for (int j = 0; j < this_field.get_Nj(); ++j, y += size_d[1])
    {
        double this_n = fabs(sign * this_field(i, j) - sin(2.0 * M_PI * (y - size_y[0] - FDTD_Const::C * t) / (size_y[1] - size_y[0])));
        if (this_n > extr_n)
            extr_n = this_n;
    }
    return extr_n;
}

void write_all(FDTD& test, char axis)
{
    std::ofstream test_fout;

#ifndef __DEBUG__
    test_fout.open("OutFile.csv");
#else
    test_fout.open("../../../PlotScript/OutFile.csv");
#endif

    if (!test_fout.is_open())
    {
        std::cout << "ERROR: Failed to open OutFile.csv" << std::endl;
        exit(1);
    }
    for (int i = static_cast<int>(Component::EX); i <= static_cast<int>(Component::BZ); ++i)
    {
        if (axis == 'x')
        {
            write_x(test.get_field(static_cast<Component>(i)), test_fout);
        }
        else write_y(test.get_field(static_cast<Component>(i)), test_fout);
    }
    test_fout.close();
}

#ifndef __DEBUG__
int main(int argc, char* argv[])
#else
int main()
#endif
{
    std::vector<double> numbers;
    std::ifstream source_fin;

#ifndef __DEBUG__
    source_fin.open("Source.txt");
#else
    source_fin.open("../../../PlotScript/Source.txt");
#endif

    if (!source_fin.is_open())
    {
        std::cout << "ERROR: Failed to open Source.txt" << std::endl;
        return 1;
    }

#ifdef __DEBUG__
    const char* argv[5] = {"0", "0", "5", "0", "0"};
#endif

    double number;
    while (source_fin >> number) 
    {
        numbers.push_back(number);
    }

    int arr_N[2] = { numbers[0], numbers[1] };
    double arr_x[2] = { numbers[2], numbers[3] };
    double arr_y[2] = { numbers[4], numbers[5] };
    double arr_d[2] = { (arr_x[1] - arr_x[0]) / arr_N[0], (arr_y[1] - arr_y[0]) / arr_N[1] };

    FDTD test_1(arr_N, arr_x, arr_y, numbers[6]);

    Component fld_1 = static_cast<Component>(std::atoi(argv[1]));
    Component fld_2 = static_cast<Component>(std::atoi(argv[2]));
    Component fld_3 = static_cast<Component>(std::atoi(argv[3]));

    bool shifted_version = std::atoi(argv[4]);
    sign = get_vector_sign(fld_1, fld_2);

    char selected_axis;
    if (fld_1 == Component::EY && fld_2 == Component::BZ || fld_1 == Component::EZ && fld_2 == Component::BY)
    {
        selected_axis = 'x';

        initial_filling_x(test_1, fld_1, fld_2, arr_N, arr_d[0], arr_x, shifted_version);

        if (shifted_version)
        {
            test_1.shifted_update_field(numbers[7]);
        }
        else test_1.update_field(numbers[7]);

        if (static_cast<int>(fld_3) > static_cast<int>(Component::EZ))
        {
            sign = 1.0;
            shift = arr_d[0] / 2;
        }
        std::cout << "Maximum absolute error: " << max_abs_error_x(test_1.get_field(fld_3), arr_N, arr_x, arr_d, numbers[7], shifted_version) << std::endl;
    }
    else if (fld_1 == Component::EX && fld_2 == Component::BZ || fld_1 == Component::EZ && fld_2 == Component::BX)
    {
        selected_axis = 'y';

        initial_filling_y(test_1, fld_1, fld_2, arr_N, arr_d[1], arr_y, shifted_version);

        if (shifted_version)
        {
            test_1.shifted_update_field(numbers[7]);
        }
        else test_1.update_field(numbers[7]);

        if (static_cast<int>(fld_3) > static_cast<int>(Component::EZ))
        {
            sign = 1.0;
            shift = arr_d[1] / 2;
        }
        std::cout << "Maximum absolute error: " << max_abs_error_y(test_1.get_field(fld_3), arr_N, arr_y, arr_d, numbers[7], shifted_version) << std::endl;
    }
    else
    {
        std::cout << "ERROR" << std::endl;
        exit(1);
    }
    write_all(test_1, selected_axis);

    source_fin.close();

    std::cout << "The work of sample.exe is completed without errors" << std::endl;

    return 0;
}
