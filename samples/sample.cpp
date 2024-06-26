﻿#define _USE_MATH_DEFINES
#define __USE_SPHERICAL_WAVE__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <cmath>

#include "test_FDTD.h"
#include "Writer.h"

Axis get_axis(Component field_E, Component field_B)
{
    Axis selected_axis;
    if (field_E == Component::EY && field_B == Component::BZ ||
        field_E == Component::EZ && field_B == Component::BY)
    {
        selected_axis = Axis::X;
    }
    else if (field_E == Component::EX && field_B == Component::BZ ||
        field_E == Component::EZ && field_B == Component::BX)
    {
        selected_axis = Axis::Y;
    }
    else if (field_E == Component::EX && field_B == Component::BY ||
        field_E == Component::EY && field_B == Component::BX)
    {
        selected_axis = Axis::Z;
    }
    else
    {
        std::cout << "ERROR: invalid selected fields" << std::endl;
        exit(1);
    }
    return selected_axis;
}

void plane_wave(std::vector<char*> arguments, 
    std::vector<double> numbers, char* outfile_path)
{
    // Saving data from the console
    Component selected_field_1 = static_cast<Component>(std::atoi(arguments[1]));
    Component selected_field_2 = static_cast<Component>(std::atoi(arguments[2]));
    Component fld_tested = static_cast<Component>(std::atoi(arguments[3]));
    bool write_flag = static_cast<bool>(std::atoi(arguments[4]));
    
    Component flds_selected[2] = { selected_field_1, selected_field_2 };

    // Saving data from the txt file
    int grid_sizes[3] = { numbers[0], numbers[1], numbers[2] };   // { Nx, Ny , Nz }
    double sizes_x[2] = { numbers[3], numbers[4] };               // { ax, bx }
    double sizes_y[2] = { numbers[5], numbers[6] };               // { ay, by }
    double sizes_z[2] = { numbers[7], numbers[8] };               // { az, bz }
    int iterations_num = static_cast<int>(numbers[9]);
    double time = numbers[10];
    double time_step = time / static_cast<double>(iterations_num);   // dt

    // Initialization of the initializing function and the true solution function
    std::function<double(double, double[2])> initial_func = [](double x, double size[2])
    {
        return sin(2.0 * M_PI * (x - size[0]) / (size[1] - size[0]));
    };

    // Determination of the wave propagation axis
    Axis axis = get_axis(flds_selected[0], flds_selected[1]);
    
    // Initialization of the structures and method
    SelectedFields current_fields
    {
        flds_selected[0],
        flds_selected[1],
    };
    
    Parameters params
    {
        grid_sizes[0],
        grid_sizes[1],
        grid_sizes[2],
        sizes_x[0],
        sizes_x[1],
        sizes_y[0],
        sizes_y[1],
        sizes_z[0],
        sizes_z[1],
        (sizes_x[1] - sizes_x[0]) / static_cast<double>(grid_sizes[0]),
        (sizes_y[1] - sizes_y[0]) / static_cast<double>(grid_sizes[1]),
        (sizes_z[1] - sizes_z[0]) / static_cast<double>(grid_sizes[2])
    };
    FDTD method(params, time_step, 0.0);
    
    // Meaningful calculations
    Test_FDTD test(params);
    try
    {
        test.initial_filling(method, current_fields, iterations_num, initial_func);
    }
    catch(const std::exception& except)
    {
        std::cout << except.what() << std::endl;
    }

    std::vector<std::vector<Field>> data = method.update_fields(iterations_num);

    if (write_flag == true)
    {
        // Writing the results to a file
        try
        {
             write_plane(method, axis, outfile_path);
        }
        catch (const std::exception& except)
        {
            std::cout << except.what() << std::endl;
        }
    }
}

void spherical_wave(int n, int it, char* base_path = "")
{
    CurrentParameters cur_param
    {
        8,
        4,
        0.2,
    };
    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;
    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);
    std::function<double(double, double, double, double)> cur_func 
        = [T, Tx, Ty, Tz](double x, double y, double z, double t)
    {
        return sin(2.0 * M_PI * t / T) 
            * pow(cos(2.0 * M_PI * x / Tx), 2.0) 
            * pow(cos(2.0 * M_PI * y / Ty), 2.0) 
            * pow(cos(2.0 * M_PI * z / Tz), 2.0);
    };
    
    // Initialization of the structures and method
    double d = FDTDconst::C;

    double boundary = static_cast<double>(n) / 2.0 * d;

    Parameters params
    {
        n,
        n,
        n,
        -boundary,
        boundary,
        -boundary,
        boundary,
        -boundary,
        boundary,
        d,
        d,
        d
    };
    FDTD method(params, cur_param.dt, 0.1);

    // Meaningful calculations
    Test_FDTD test(params);
    test.initiialize_current(method, cur_param, it, cur_func);
    std::vector<std::vector<Field>> data = method.update_fields(it);
    try
    {
        write_spherical(data, Axis::X, it, base_path);
    }
    catch (const std::exception& except)
    {
        std::cout << except.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    std::ifstream source_fin;
    char* outfile_path;
    
    std::vector<char*> arguments(argv, argv + argc);
    if (argc == 1) 
    {
#ifdef __USE_SPHERICAL_WAVE__
        int N = 70;
        int Iterations = 410;
        spherical_wave(N, Iterations, "../../PlotScript/");
#endif

#ifndef __USE_SPHERICAL_WAVE__
        source_fin.open("../../PlotScript/Source.txt");
        outfile_path = "../../PlotScript/OutFile.csv";

        struct StringComponent {
            char* EX = "0";
            char* EY = "1";
            char* EZ = "2";
            char* BX = "3";
            char* BY = "4";
            char* BZ = "5";
        } field;
        const size_t size_tmp = 5;
        char* tmp[size_tmp]{ "1", field.EX, field.BZ, field.BZ, "1" };
        
        arguments.clear();
        arguments.reserve(size_tmp);
        std::copy(std::begin(tmp), std::end(tmp), std::back_inserter(arguments));

        if (!source_fin.is_open())
        {
            std::cout << "ERROR: Failed to open Source.txt" << std::endl;
            return 1;
        }
        std::vector<double> numbers;
        double number;
        while (source_fin >> number)
        {
            numbers.push_back(number);
        }
        source_fin.close();

        plane_wave(arguments, numbers, outfile_path);
#endif
    }
    else if (argc == 4)
    {
        int N = std::atoi(arguments[1]);
        int Iterations = std::atoi(arguments[2]);
        spherical_wave(N, Iterations);
    }
    else
    {
        source_fin.open("Source.txt");
        outfile_path = "OutFile.csv";
        if (!source_fin.is_open())
        {
            std::cout << "ERROR: Failed to open Source.txt" << std::endl;
            return 1;
        }
        std::vector<double> numbers;
        double number;
        while (source_fin >> number)
        {
            numbers.push_back(number);
        }
        plane_wave(arguments, numbers, outfile_path);
        source_fin.close();
    }
    return 0;
}
