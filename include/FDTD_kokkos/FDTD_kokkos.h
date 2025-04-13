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

    TimeField Jx, Jy, Jz;

    Field Exy, Exz, Eyx, Eyz, Ezx, Ezy;
    Field Bxy, Bxz, Byx, Byz, Bzx, Bzy;

    // Permittivity and permeability of the medium
    Field EsigmaX, EsigmaY, EsigmaZ, 
          BsigmaX, BsigmaY, BsigmaZ;

    Parameters parameters;
    CurrentParameters cParams;
    double pml_percent, dt;
    int pml_size_i, pml_size_j, pml_size_k, time;
    std::function<double(double, double, double, double)> init_func;

    void update_B_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]);
    void update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]);
    void write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it);

public:
    FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar = CurrentParameters(),
    std::function<double(double, double, double, double)> init_function = nullptr);

    Field& get_field(Component);

    void update_fields(bool write_result = false, 
        Axis write_axis = Axis::X, std::string base_path = "");
};

}
