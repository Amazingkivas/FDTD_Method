#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>
#include <cstdlib>


#include "FDTD_kokkos.h"

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

void spherical_wave(int n, int it, const std::string base_path = "../../PlotScript/")
{
    CurrentParameters cur_param
    {
        8,
        4,
        0.2
    };

    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;

    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);

    std::function<double(double, double, double, double)> cur_func =
        [T, Tx, Ty, Tz](double x, double y, double z, double t)
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

    FDTD_kokkos::FDTD method(params, cur_param.dt, 0.0, it, cur_param, cur_func);

    auto start = std::chrono::high_resolution_clock::now();
    method.update_fields(false, Axis::Z, base_path);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        if (Kokkos::hwloc::available())
            std::cout << "hwloc available" << std::endl;
        std::ostringstream msg;
        msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
        Kokkos::print_configuration(msg);
        std::cout << msg.str();


        std::ifstream source_fin;
        std::vector<char*> arguments(argv, argv + argc);

        if (argc == 1)
        {
            int N = 512;
            int Iterations = 25;
            spherical_wave(N, Iterations, "../../");
        }
        else if (argc == 4)
        {
            int N = std::atoi(arguments[1]);
            int Iterations = std::atoi(arguments[2]);
            spherical_wave(N, Iterations, "");
        }
        else
        {
            std::cout << "ERROR: Incorrect number of parameters" << std::endl;
            exit(1);
        }
    }
    Kokkos::finalize();
    return 0;
}
