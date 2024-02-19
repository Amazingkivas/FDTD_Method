#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

namespace FDTD_Const
{
    const double C = 3e10;
}

class Field
{
private:
    int Ni;
    int Nj;
    int Nk;
    std::vector<double> field;

public:
    Field(const int, const int, const int);
    Field& operator= (const Field& other);

    double& operator() (int _i, int _j, int _k);

    int get_Ni() { return Ni; }
    int get_Nj() { return Nj; }
    int get_Nk() { return Nk; }
};

enum class Component { EX, EY, EZ, BX, BY, BZ };

class FDTD
{
private:
    Field Ex, Ey, Ez, Bx, By, Bz;
    int Ni, Nj, Nk;
    double ax, bx, ay, by, az, bz, dx, dy, dz, dt;

public:
    FDTD(int size_grid[3], double size_x[2], double size_y[2], double size_z[2], double _dt);

    Field& get_field(Component);

    //void update_field(const int);
    void shifted_update_field(const int);

    int get_Ni() { return Ni; }
    int get_Nj() { return Nj; }
    int get_Nk() { return Nk; }
};
