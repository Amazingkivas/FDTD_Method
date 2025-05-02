#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>

#include "kokkos_functors.h"

using namespace FDTD_struct;

namespace FDTD_kokkos
{
class FDTD
{
private:
    Field Ex, Ey, Ez, Bx, By, Bz;

    Field Jx, Jy, Jz;

    Parameters parameters;
    CurrentParameters cParams;
    double pml_percent, dt;
    int pml_size_i, pml_size_j, pml_size_k, time;
    std::function<double(double, double, double, double)> init_func;

public:
    FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar = CurrentParameters(),
    std::function<double(double, double, double, double)> init_function = nullptr);

    Field& get_field(Component);

    void update_fields(bool write_result = false, 
        Axis write_axis = Axis::X, std::string base_path = "");
};

}
