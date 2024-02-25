#pragma once

#define _USE_MATH_DEFINES

#include <vector>
#include <omp.h>

#include "Structures.h"

using namespace FDTDstruct;

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
    std::vector<Field> Jx;
    std::vector<Field> Jy;
    std::vector<Field> Jz;
    Parameters parameters;
    double dt;

public:
    FDTD(Parameters _parameters, double _dt);

    Field& get_field(Component);
    std::vector<Field>& get_current(Component);

    void update_fields(const int time);
};
