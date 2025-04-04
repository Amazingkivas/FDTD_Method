#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>

#include "test_FDTD.h"

using namespace FDTD_struct;

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

void spherical_wave(int n, int it, std::string base_path = "")
{
    double dt = 0.2;
    CurrentParameters cur_param
    {
        8,
        4,
    };

    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;
    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / dt);

    std::function<double(double, double, double, double)> cur_func 
        = [T, Tx, Ty, Tz](double x, double y, double z, double t)
    {
        return sin(2.0 * M_PI * t / T) 
            * pow(cos(2.0 * M_PI * x / Tx), 2.0) 
            * pow(cos(2.0 * M_PI * y / Ty), 2.0) 
            * pow(cos(2.0 * M_PI * z / Tz), 2.0);
    };
    
    double d = FDTD_const::C;
    double boundary = static_cast<double>(n) / 2.0 * d;

    Parameters params
    {
        n,          // Ni
        n,          // Nj
        n,          // Nk
        dt,
        -boundary,  // x_min
        boundary,   // x_max
        -boundary,  // y_min
        boundary,   // y_max
        -boundary,  // z_min
        boundary,   // z_max
        d,          // dx
        d,          // dy
        d           // dz
    };

    FDTD_openmp::FDTD method(params);
    int first_iters = std::min(it, cur_param.iterations);

    double dx = params.dx;
    double dy = params.dy;
    double dz = params.dz;

    int start_i = static_cast<int>(std::floor((-cur_param.period_x / 4.0 - params.ax) / dx));
    int start_j = static_cast<int>(std::floor((-cur_param.period_y / 4.0 - params.ay) / dy));
    int start_k = static_cast<int>(std::floor((-cur_param.period_z / 4.0 - params.az) / dz));

    int max_i = static_cast<int>(std::floor((cur_param.period_x / 4.0 - params.ax) / dx));
    int max_j = static_cast<int>(std::floor((cur_param.period_y / 4.0 - params.ay) / dy));
    int max_k = static_cast<int>(std::floor((cur_param.period_z / 4.0 - params.az) / dz));
    
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < first_iters; iter++) {
        for (int k = start_k; k < max_k; ++k) {
            for (int j = start_j; j < max_j; ++j) {
                for (int i = start_i; i < max_i; ++i) {
                    int index = i + j * params.Ni + k * params.Ni * params.Nj;

                    double J_value = cur_func(i * dx, j * dy, k * dz, (iter + 1) * params.dt);

                    method.get_field(Component::JX)[index] = J_value;
                    method.get_field(Component::JY)[index] = J_value;
                    method.get_field(Component::JZ)[index] = J_value;
                }
            }
        }
        method.update_fields();
    }
    method.zeroes_currents();

    for (int iter = first_iters; iter < it; iter++)
    {
        method.update_fields();
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;
}

int main(int argc, char* argv[])
{
    std::ifstream source_fin;
    std::vector<char*> arguments(argv, argv + argc);
    if (argc == 1) 
    {
        int N = 120;
        int Iterations = 50;
        spherical_wave(N, Iterations, "../../");
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
