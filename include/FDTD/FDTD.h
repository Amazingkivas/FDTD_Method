#pragma once

#define _USE_MATH_DEFINES

#include <vector>
#include <omp.h>
#include <functional>
#include <iostream>
#include <cmath>

#include "Writer.h"
#include "Field.h"
#include "Current.h"

#include "FDTD_boundaries.h"

using namespace FDTDstruct;

class FDTD
{
private:
    Field Ex, Ey, Ez, Bx, By, Bz;

    Current Jx, Jy, Jz;

    Field Exy, Exz, Eyx, Eyz, Ezx, Ezy;
    Field Bxy, Bxz, Byx, Byz, Bzx, Bzy;

    // Permittivity and permeability of the medium
    Field EsigmaX, EsigmaY, EsigmaZ, 
          BsigmaX, BsigmaY, BsigmaZ;

    Parameters parameters;
    double pml_percent, dt;
    int pml_size_i, pml_size_j, pml_size_k;

    void applyPeriodicBoundary(int& i, int& j, int& k, int Ni, int Nj, int Nk);

    void update_E(int bounds_i[2], int bounds_j[2], int bounds_k[2], int t);
    void update_B(int bounds_i[2], int bounds_j[2], int bounds_k[2]);

    double PMLcoef(const double& sigma);
    void update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]);
    void update_B_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]);

    void set_sigma_x(int bounds_i[2], int bounds_j[2], int bounds_k[2],
        double SGm, std::function<int(int, int, int)> dist);
    void set_sigma_y(int bounds_i[2], int bounds_j[2], int bounds_k[2],
        double SGm, std::function<int(int, int, int)> dist);
    void set_sigma_z(int bounds_i[2], int bounds_j[2], int bounds_k[2],
        double SGm, std::function<int(int, int, int)> dist);

public:
    FDTD(Parameters _parameters, double _dt, double _pml_percent, int current_iters = 1);

    Field& get_field(Component);
    Current& get_current(Component);

    std::vector<Field> update_fields(const int time, bool write_result = false, 
        Axis write_axis = Axis::X, std::string base_path = "");
};
