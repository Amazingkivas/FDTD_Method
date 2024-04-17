#pragma once

#define _USE_MATH_DEFINES

#include <vector>
#include <omp.h>
#include <functional>

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

    Field Exy, Exz, Eyx, Eyz, Ezx, Ezy;
    Field Bxy, Bxz, Byx, Byz, Bzx, Bzy;
    Field EsigmaX, EsigmaY, EsigmaZ, 
          BsigmaX, BsigmaY, BsigmaZ;

    Parameters parameters;
    double pml_percent, dt;
    int pml_size_i, pml_size_j, pml_size_k;

    void update_E(int bounds_i[2], int bounds_j[2], int bounds_k[2], int t);
    void update_B(int bounds_i[2], int bounds_j[2], int bounds_k[2]);

    double PMLcoef(double sigma);
    void update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]);
    void update_B_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]);

    void set_sigma_x(int bounds_i[2], int bounds_j[2], int bounds_k[2],
        double SGm, std::function<int(int, int, int)> dist);
    void set_sigma_y(int bounds_i[2], int bounds_j[2], int bounds_k[2],
        double SGm, std::function<int(int, int, int)> dist);
    void set_sigma_z(int bounds_i[2], int bounds_j[2], int bounds_k[2],
        double SGm, std::function<int(int, int, int)> dist);

    double calc_max_value(Field& field);

public:
    FDTD(Parameters _parameters, double _dt, double _pml_percent);

    Field& get_field(Component);
    std::vector<Field>& get_current(Component);

    std::vector<std::vector<Field>> update_fields(const int time);

    double calc_reflection(Field E_start[3], Field B_start[3],
        Field E_final[3], Field B_final[3]);
};
