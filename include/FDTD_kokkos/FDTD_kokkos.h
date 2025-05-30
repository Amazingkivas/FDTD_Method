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
protected:
    Field Jx, Jy, Jz;
    Field Ex, Ey, Ez;
    Field Bx, By, Bz;

    int Ni, Nj, Nk;
    double dx, dy, dz, dt;
    double current_coef;
    double coef_Ex, coef_Ey, coef_Ez;
    double coef_Bx, coef_By, coef_Bz;
    int begin_main_i, begin_main_j, begin_main_k;
    int end_main_i, end_main_j, end_main_k;

    Parameters parameters;

public:
    FDTD(Parameters _parameters, double _dt);

    Field& get_field(Component);

    virtual void update_fields();

    void zeroed_currents();
};

}
