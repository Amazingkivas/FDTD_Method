#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include "FDTD.h"

void initial_filling(FDTD& test, int size_N[2], double size_d[2], double size_x[2], double size_y[2])
{
    double x = size_d[0];
    for (int i = 0; i < size_N[0]; x += size_d[0], ++i)
    {
        for (int j = 0; j < size_N[1]; ++j)
        {
            test.get_field(Component::EY)(i, j) = sin(2 * M_PI * (x - size_x[0]) / (size_x[1] - size_x[0]));
            test.get_field(Component::BZ)(i, j) = sin(2 * M_PI * (x - size_x[0]) / (size_x[1] - size_x[0]));
        }
    }
}

void write(Field& this_field, std::ofstream& fout)
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


double min_abs_error(Field& this_field, int size_N[2], double size_x[2], double size_d[2], double t)
{
    double extr_n = 0.0;
    double x = 0;
    int j = 0;
    for (int i = 0; i < this_field.get_Ni(); ++i, x += size_d[0])
    {
        double this_n = fabs(fabs(this_field(i, j)) - fabs(sin(2 * M_PI * (x - size_x[0] - FDTD_Const::C * t) / (size_x[1] - size_x[0]))));
        if (this_n > extr_n)
            extr_n = this_n;
    }
    return extr_n;
}


void write_all(FDTD& test)
{
    std::ofstream test_fout;
    test_fout.open("OutFile.csv");
    if (!test_fout.is_open())
    {
        std::cout << "ERROR: Failed to open the file!" << std::endl;
        return;
    }
    write(test.get_field(Component::EX), test_fout);
    write(test.get_field(Component::EY), test_fout);
    write(test.get_field(Component::EZ), test_fout);
    write(test.get_field(Component::BX), test_fout);
    write(test.get_field(Component::BY), test_fout);
    write(test.get_field(Component::BZ), test_fout);
    
    test_fout.close();
}

int main()
{
    std::vector<double> numbers;
    std::ifstream source_fin;
    source_fin.open("Source.txt");

    if (!source_fin.is_open()) 
    {
        std::cout << "ERROR: Failed to open the file!!" << std::endl;
        return 1;
    }

    double number;
    while (source_fin >> number) {
        numbers.push_back(number);
    }
    
    int arr_N[2] = { numbers[0], numbers[1] };
    double arr_x[2] = { numbers[2], numbers[3] };
    double arr_y[2] = { numbers[4], numbers[5] };
    double arr_d[2] = { (arr_x[1] - arr_x[0]) / arr_N[0], (arr_y[1] - arr_y[0]) / arr_N[1] };
    
    FDTD test_1(arr_N, arr_x, arr_y, numbers[6]);

    initial_filling(test_1, arr_N, arr_d, arr_x, arr_y);
    test_1.update_field(numbers[7]);
    
    std::cout << "Minimum absolute error: " << min_abs_error(test_1.get_field(Component::EY), arr_N, arr_x, arr_d, numbers[7]) << std::endl;
    write_all(test_1);

    source_fin.close();

    std::cout << "The work of sample.exe is completed without errors" << std::endl;

    return 0;
}
