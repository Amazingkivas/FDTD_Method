#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>

namespace fs = std::filesystem;

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

void spherical_wave(int n, int it, std::string base_path = "")
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
    double d = FDTD_const::C;

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

    auto clear_directory = [](const std::string& dir_path) {
        if (fs::exists(dir_path) && fs::is_directory(dir_path)) {
            for (auto& file : fs::directory_iterator(dir_path)) {
                if (fs::is_regular_file(file.path())) {
                    fs::remove(file.path());
                }
            }
        } else {
            fs::create_directories(dir_path);
        }
    };

    for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
    {
        std::string dir_path = base_path + "OutFiles_" + std::to_string(c + 1) + "/";
        clear_directory(dir_path);
    }

    FDTD method(params, cur_param.dt, 0.0, it, cur_param, cur_func);
    
    auto start = std::chrono::high_resolution_clock::now();
    method.update_fields(false, Axis::Z, base_path);
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
        int N = 50;
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
