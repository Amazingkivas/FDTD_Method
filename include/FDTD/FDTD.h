#pragma once

#define _USE_MATH_DEFINES

#include <vector>
#include <omp.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

#include "FDTD_boundaries.h"
#include "functors.h"

using namespace FDTD_struct;

class FDTD {
public:
FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar = {},
    std::function<double(double, double, double, double)> init_function = [](double a, double b, double c, double d){return 0.0;});

    Field& get_field(Component this_field);
    void update_fields(bool write_result = false, Axis write_axis = Axis::X, std::string base_path = "");
    
private:
    Parameters parameters;
    CurrentParameters cParams;
    double dt;
    double pml_percent;
    int time;
    std::function<double(double, double, double, double)> init_func;

    Field Jx, Jy, Jz;
    Field Ex, Ey, Ez;
    Field Bx, By, Bz;

    int pml_size_i, pml_size_j, pml_size_k;

    void applyPeriodicBoundaryB();
    void applyPeriodicBoundaryE();
};

