#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "Structures.h"

using namespace FDTDstruct;

namespace FDTDconst
{
    const double C = 3e10;  // light speed
}

class Field
{
private:
    int Ni, Nj, Nk;
    std::vector<double> field;

public:
    Field(const int _Ni, const int _Nj, const int _Nk);
    Field& operator= (const Field& other);

    double& operator() (int i, int j, int k);

    int get_Ni() { return Ni; }
    int get_Nj() { return Nj; }
    int get_Nk() { return Nk; }
};

class FDTD
{
private:
    Field Ex, Ey, Ez, Bx, By, Bz;
    Parameters parameters;
    double dt;

public:
    FDTD(Parameters _parameters, double _dt);

    Field& get_field(Component);

    //void update_field(const int);
    void shifted_update_field(const int time);
};
