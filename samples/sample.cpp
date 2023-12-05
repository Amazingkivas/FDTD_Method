#define _USE_MATH_DEFINES
#define __DEBUG__

#include <iostream>
#include <iomanip>
#include <fstream>

#include "test_FDTD.h"

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

void write_all(FDTD& test, char axis)
{
    std::ofstream test_fout;

#ifndef __DEBUG__
    test_fout.open("OutFile.csv");
#else
    test_fout.open("../../PlotScript/OutFile.csv");
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
    source_fin.open("../../PlotScript/Source.txt");
#endif

    if (!source_fin.is_open())
    {
        std::cout << "ERROR: Failed to open Source.txt" << std::endl;
        return 1;
    }

#ifdef __DEBUG__
    const char* argv[5] = {"1", "2", "3", "2", "1"};
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

    double courant_condtition = 0.25;
    double dt;
    double t;

    Component flds_selected[2] = { static_cast<Component>(std::atoi(argv[1])), static_cast<Component>(std::atoi(argv[2])) };
    Component fld_tested = static_cast<Component>(std::atoi(argv[3]));
    bool shifted_flag = static_cast<bool>(std::atoi(argv[4]));


    std::function<double(double, double[2])> initial_func =
        [](double x, double size[2]) { return sin(2.0 * M_PI * (x - size[0]) / (size[1] - size[0])); };

    std::function<double(double, double, double[2])> true_func =
        [](double x, double t, double size[2]) { return sin(2.0 * M_PI * (x - size[0] - FDTD_Const::C * t) / (size[1] - size[0])); };


    char selected_axis;
    if (flds_selected[0] == Component::EY && flds_selected[1] == Component::BZ ||
        flds_selected[0] == Component::EZ && flds_selected[1] == Component::BY)
    {
        selected_axis = 'x';
        dt = courant_condtition * arr_d[0] / FDTD_Const::C;
        t = courant_condtition * (arr_x[1] - arr_x[0]) / FDTD_Const::C;
    }
    else if (flds_selected[0] == Component::EX && flds_selected[1] == Component::BZ ||
        flds_selected[0] == Component::EZ && flds_selected[1] == Component::BX)
    {
        selected_axis = 'y';
        dt = courant_condtition * arr_d[1] / FDTD_Const::C;
        t = courant_condtition * (arr_y[1] - arr_y[0]) / FDTD_Const::C;
    }
    else
    {
        std::cout << "ERROR" << std::endl;
        exit(1);
    }


    FDTD test_1(arr_N, arr_x, arr_y, dt);


    int iter_nums = static_cast<int>(numbers[6]);
    Test_FDTD test(test_1, flds_selected, fld_tested, arr_x, arr_y, arr_d, t, iter_nums, initial_func, true_func, shifted_flag);
    std::cout << test.get_max_abs_error() << std::endl;


    write_all(test_1, selected_axis);

    source_fin.close();

    return 0;
}
