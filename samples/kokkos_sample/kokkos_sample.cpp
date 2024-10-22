#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <cmath>
#include <Kokkos_Core.hpp>
#include "FDTD_kokkos.h"

// Функция для определения оси
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

// Основная функция для моделирования сферической волны
void spherical_wave(int n, int it, const char* base_path = "../../PlotScript/")
{
    // Параметры тока
    CurrentParameters cur_param
    {
        8,     // period
        4,     // period_x
        0.2    // dt
    };

    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;

    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);

    // Функция для тока
    std::function<double(double, double, double, double)> cur_func =
        [T, Tx, Ty, Tz](double x, double y, double z, double t)
    {
        return sin(2.0 * M_PI * t / T)
            * pow(cos(2.0 * M_PI * x / Tx), 2.0)
            * pow(cos(2.0 * M_PI * y / Ty), 2.0)
            * pow(cos(2.0 * M_PI * z / Tz), 2.0);
    };

    // Инициализация параметров для FDTD
    double d = FDTDconst::C;
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

    FDTD method(params, cur_param, cur_param.dt, 0.0, cur_func, it);

    // Выполнение обновления полей
    method.update_fields(true, Axis::Z, base_path);
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);  // Инициализация Kokkos
    {
        std::ifstream source_fin;
        std::vector<char*> arguments(argv, argv + argc);

        // Выполнение с параметрами по умолчанию
        if (argc == 1)
        {
            int N = 70;
            int Iterations = 110;
            spherical_wave(N, Iterations, "../../");
        }
        // Выполнение с пользовательскими параметрами
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
    Kokkos::finalize();  // Завершение Kokkos
    return 0;
}