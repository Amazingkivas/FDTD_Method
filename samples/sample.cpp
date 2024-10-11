#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <cmath>

#include "test_FDTD.h"

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
    FDTD method(params, cur_param.dt, 0.1, cur_param.iterations);

    // Meaningful calculations
    Test_FDTD test(params);
    test.initiialize_current(method, cur_param, it, cur_func);
    method.update_fields(it, true, Axis::X, "");
}

int main(int argc, char* argv[])
{
    std::ifstream source_fin;
    char* outfile_path;
    
    std::vector<char*> arguments(argv, argv + argc);
    if (argc == 1) 
    {
        int N = 70;
        int Iterations = 410;
        spherical_wave(N, Iterations, "../../PlotScript/");
    }
    else if (argc == 4)
    {
        int N = std::atoi(arguments[1]);
        int Iterations = std::atoi(arguments[2]);
        spherical_wave(N, Iterations);
    }
    else
    {
        std::cout << "ERROR: Incorrect number of parameters" << std::endl;
        exit(1);
    }
    return 0;
}
