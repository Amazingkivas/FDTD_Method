#pragma once

#define _USE_MATH_DEFINES

#include <Kokkos_Core.hpp>

#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

#include "Structures.h"
#include "Writer_kokkos.h"

using namespace FDTDstruct;

using Device = Kokkos::DefaultExecutionSpace;
using Field = Kokkos::View<double***, Device>;
using TimeField = Kokkos::View<double****, Device>;
using Function = std::function<int(int, int, int)>;

void applyPeriodicBoundary(int& i, int& j, int& k, int Ni, int Nj, int Nk)
{
    int i_isMinusOne = (i < 0);
	int j_isMinusOne = (j < 0);
	int k_isMinusOne = (k < 0);

	int i_isNi = (i == Ni);
	int j_isNj = (j == Nj);
	int k_isNk = (k == Nk);

	i = (Ni - 1) * i_isMinusOne + i *
		!(i_isMinusOne || i_isNi);
	j = (Nj - 1) * j_isMinusOne + j *
		!(j_isMinusOne || j_isNj);
	k = (Nk - 1) * k_isMinusOne + k *
		!(k_isMinusOne || k_isNk);

}

// double PMLcoef(double sigma)
// {
//     return std::exp(-sigma * dt * FDTDconst::C);
// }

// class ComputeSigmaFunctor
// {
// private:
//     double SGm;
//     Function dist;
//     int pml_size;
//     Field EsigmaX, BsigmaX;
// public:
//     ComputeSigmaXFunctor(Field _EsigmaX, Field _BsigmaX, double _SGm, Function distance, int _pml_size) :
//        EsigmaX(_EsigmaX), BsigmaX(_BsigmaX), SGm(_SGm), dist(distance), pml_size(_pml_size) {};

//     static void apply(Field _EsigmaX, Field _BsigmaX, double _SGm, Function distance, int _pml_size,
//        int bounds_i[2], int bounds_j[2], int bounds_k[2])
//     {
//         ComputeSigmaXFunctor functor(_EsigmaX, _BsigmaX, _SGm, distance, _pml_size);
//         Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
//             { bounds_i[0], bounds_j[0], bounds_k[0] }, { bounds_i[1], bounds_j[1], bounds_k[1] }),
//             functor);
//     }

//     KOKKOS_INLINE_FUNCTION void operator()(const int& i, const int& j, const int& k) const
//     {
//         EsigmaX(i, j, k) = SGm * std::pow(static_cast<double>(dist(i, j, k))
//            / static_cast<double>(pml_size), FDTDconst::N);
//         BsigmaX(i, j, k) = SGm * std::pow(static_cast<double>(dist(i, j, k))
//            / static_cast<double>(pml_size), FDTDconst::N);
//     }
// };

class ComputeE_FieldFunctor
{
private:
    Field Ex, Ey, Ez;
    Field Bx, By, Bz;
    TimeField Jx, Jy, Jz;
    double dt, dx, dy, dz;
    int t, iters;
    int Ni, Nj, Nk;
public:
    ComputeE_FieldFunctor(Field Ex, Field Ey, Field Ez,
                        Field Bx, Field By, Field Bz,
                        TimeField Jx, TimeField Jy, TimeField Jz,
                        double dt, double dx, double dy, double dz, int t, int iters,
                        int Ni, int Nj, int Nk)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz),
          Jx(Jx), Jy(Jy), Jz(Jz), dt(dt), dx(dx), dy(dy), dz(dz), t(t), iters(iters), 
          Ni(Ni), Nj(Nj), Nk(Nk) {}

    static void apply(Field Ex, Field Ey, Field Ez,
                      Field Bx, Field By, Field Bz,
                      TimeField Jx, TimeField Jy, TimeField Jz,
                      double dt, double dx, double dy, double dz,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2], int t, int iters,
                      int Ni, int Nj, int Nk) {
        
        ComputeE_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dt, dx, dy, dz, t, iters, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_i[0], bounds_j[0], bounds_k[0]},
                                                      {bounds_i[1], bounds_j[1], bounds_k[1]});

        Kokkos::parallel_for("UpdateEField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const 
    {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

        double Jx_val = (iters <= t) ? 0.0 : 4.0 * FDTDconst::PI * dt * Jx(t, i, j, k);
        double Jy_val = (iters <= t) ? 0.0 : 4.0 * FDTDconst::PI * dt * Jy(t, i, j, k);
        double Jz_val = (iters <= t) ? 0.0 : 4.0 * FDTDconst::PI * dt * Jz(t, i, j, k);

        Ex(i, j, k) += -Jx_val + FDTDconst::C * dt * ((Bz(i, j, k) - Bz(i, j_pred, k)) / dy -
                                (By(i, j, k) - By(i, j, k_pred)) / dz);
        Ey(i, j, k) += -Jy_val + FDTDconst::C * dt * ((Bx(i, j, k) - Bx(i, j, k_pred)) / dz -
                                (Bz(i, j, k) - Bz(i_pred, j, k)) / dx);
        Ez(i, j, k) += -Jz_val + FDTDconst::C * dt * ((By(i, j, k) - By(i_pred, j, k)) / dx -
                                (Bx(i, j, k) - Bx(i, j_pred, k)) / dy);
    }
};

class ComputeB_FieldFunctor
{
private:
    Field Ex, Ey, Ez;
    Field Bx, By, Bz;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;
public:
    ComputeB_FieldFunctor(Field Ex, Field Ey, Field Ez,
                        Field Bx, Field By, Field Bz,
                        double dt, double dx, double dy, double dz,
                        int Ni, int Nj, int Nk)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz), dt(dt), dx(dx), dy(dy), dz(dz), 
        Ni(Ni), Nj(Nj), Nk(Nk) {}

    static void apply(Field Ex, Field Ey, Field Ez,
                      Field Bx, Field By, Field Bz,
                      double dt, double dx, double dy, double dz,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2], int Ni, int Nj, int Nk) {
        
        ComputeB_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_i[0], bounds_j[0], bounds_k[0]},
                                                      {bounds_i[1], bounds_j[1], bounds_k[1]});

        Kokkos::parallel_for("UpdateBField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const 
    {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

        Bx(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ey(i, j, k_next) - Ey(i, j, k)) / dz -
                        (Ez(i, j_next, k) - Ez(i, j, k)) / dy);
        By(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ez(i_next, j, k) - Ez(i, j, k)) / dx -
                        (Ex(i, j, k_next) - Ex(i, j, k)) / dz);
        Bz(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ex(i, j_next, k) - Ex(i, j, k)) / dy -
                        (Ey(i_next, j, k) - Ey(i, j, k)) / dx);
    }
};

class ComputeE_PML_FieldFunctor
{
private:
    Field Ex, Ey, Ez;
    Field Exy, Eyx, Ezy, Eyz, Ezx, Exz;
    Field Bx, By, Bz;
    Field EsigmaX, EsigmaY, EsigmaZ;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;

    double PMLcoef(double sigma) const
    {
        return std::exp(-sigma * dt * FDTDconst::C);
    }
public:
    ComputeE_PML_FieldFunctor(Field Ex, Field Ey, Field Ez,
                      Field Exy, Field Eyx, Field Ezy,
                      Field Eyz, Field Ezx, Field Exz,
                      Field Bx, Field By, Field Bz,
                      Field EsigmaX, Field EsigmaY, Field EsigmaZ,
                      double dt, double dx, double dy, double dz,
                      int Ni, int Nj, int Nk)
        : Ex(Ex), Ey(Ey), Ez(Ez), Exy(Exy), Eyx(Eyx), Ezy(Ezy),
          Eyz(Eyz), Ezx(Ezx), Exz(Exz), Bx(Bx), By(By), Bz(Bz),
          EsigmaX(EsigmaX), EsigmaY(EsigmaY), EsigmaZ(EsigmaZ),
          dt(dt), dx(dx), dy(dy), dz(dz), Ni(Ni), Nj(Nj), Nk(Nk) {}

    static void apply(Field Ex, Field Ey, Field Ez,
                      Field Exy, Field Eyx, Field Ezy,
                      Field Eyz, Field Ezx, Field Exz,
                      Field Bx, Field By, Field Bz,
                      Field EsigmaX, Field EsigmaY, Field EsigmaZ,
                      double dt, double dx, double dy, double dz,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2],
                      int Ni, int Nj, int Nk) {
        
        ComputeE_PML_FieldFunctor functor(Ex, Ey, Ez, Exy, Eyx, Ezy, Eyz, Ezx, Exz,
                                   Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
                                   dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_i[0], bounds_j[0], bounds_k[0]},
                                                      {bounds_i[1], bounds_j[1], bounds_k[1]});

        Kokkos::parallel_for("UpdateEPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const 
    {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        if (EsigmaX(i, j, k) != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(EsigmaX(i, j, k))) / (EsigmaX(i, j, k) * dx);
        else
            PMLcoef2_x = FDTDconst::C * dt / dx;

        if (EsigmaY(i, j, k) != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(EsigmaY(i, j, k))) / (EsigmaY(i, j, k) * dy);
        else
            PMLcoef2_y = FDTDconst::C * dt / dy;

        if (EsigmaZ(i, j, k) != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ(i, j, k))) / (EsigmaZ(i, j, k) * dz);
        else
            PMLcoef2_z = FDTDconst::C * dt / dz;

        Eyx(i, j, k) = Eyx(i, j, k) * PMLcoef(EsigmaX(i, j, k)) -
                    PMLcoef2_x * (Bz(i, j, k) - Bz(i_pred, j, k));
        Ezx(i, j, k) = Ezx(i, j, k) * PMLcoef(EsigmaX(i, j, k)) +
                    PMLcoef2_x * (By(i, j, k) - By(i_pred, j, k));

        Exy(i, j, k) = Exy(i, j, k) * PMLcoef(EsigmaY(i, j, k)) +
                    PMLcoef2_y * (Bz(i, j, k) - Bz(i, j_pred, k));
        Ezy(i, j, k) = Ezy(i, j, k) * PMLcoef(EsigmaY(i, j, k)) -
                    PMLcoef2_y * (Bx(i, j, k) - Bx(i, j_pred, k));

        Exz(i, j, k) = Exz(i, j, k) * PMLcoef(EsigmaZ(i, j, k)) -
                    PMLcoef2_z * (By(i, j, k) - By(i, j, k_pred));
        Eyz(i, j, k) = Eyz(i, j, k) * PMLcoef(EsigmaZ(i, j, k)) +
                    PMLcoef2_z * (Bx(i, j, k) - Bx(i, j, k_pred));

        Ex(i, j, k) = Exz(i, j, k) + Exy(i, j, k);
        Ey(i, j, k) = Eyx(i, j, k) + Eyz(i, j, k);
        Ez(i, j, k) = Ezy(i, j, k) + Ezx(i, j, k);
    }
};

class ComputeB_PML_FieldFunctor
{
private:
    Field Ex, Ey, Ez;
    Field Bxy, Byx, Bzy, Byz, Bzx, Bxz;
    Field Bx, By, Bz;
    Field BsigmaX, BsigmaY, BsigmaZ;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;

    double PMLcoef(double sigma) const
    {
        return std::exp(-sigma * dt * FDTDconst::C);
    }
public:
    ComputeB_PML_FieldFunctor(Field Ex, Field Ey, Field Ez,
                      Field Bxy, Field Byx, Field Bzy,
                      Field Byz, Field Bzx, Field Bxz,
                      Field Bx, Field By, Field Bz,
                      Field EsigmaX, Field EsigmaY, Field EsigmaZ,
                      double dt, double dx, double dy, double dz,
                      int Ni, int Nj, int Nk)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bxy(Bxy), Byx(Byx), Bzy(Bzy),
          Byz(Byz), Bzx(Bzx), Bxz(Bxz), Bx(Bx), By(By), Bz(Bz),
          BsigmaX(BsigmaX), BsigmaY(BsigmaY), BsigmaZ(BsigmaZ),
          dt(dt), dx(dx), dy(dy), dz(dz), Ni(Ni), Nj(Nj), Nk(Nk) {}

    static void apply(Field Ex, Field Ey, Field Ez,
                      Field Bxy, Field Byx, Field Bzy,
                      Field Byz, Field Bzx, Field Bxz,
                      Field Bx, Field By, Field Bz,
                      Field BsigmaX, Field BsigmaY, Field BsigmaZ,
                      double dt, double dx, double dy, double dz,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2],
                      int Ni, int Nj, int Nk) {
        
        ComputeB_PML_FieldFunctor functor(Ex, Ey, Ez, Bxy, Byx, Bzy, Byz, Bzx, Bxz,
                                   Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
                                   dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_i[0], bounds_j[0], bounds_k[0]},
                                                      {bounds_i[1], bounds_j[1], bounds_k[1]});

        Kokkos::parallel_for("UpdateBPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const 
    {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        if (BsigmaX(i, j, k) != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(BsigmaX(i, j, k))) / (BsigmaX(i, j, k) * dx);
        else
            PMLcoef2_x = FDTDconst::C * dt / dx;

        if (BsigmaY(i, j, k) != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(BsigmaY(i, j, k))) / (BsigmaY(i, j, k) * dy);
        else
            PMLcoef2_y = FDTDconst::C * dt / dy;

        if (BsigmaZ(i, j, k) != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ(i, j, k))) / (BsigmaZ(i, j, k) * dz);
        else
            PMLcoef2_z = FDTDconst::C * dt / dz;

        Byx(i, j, k) = Byx(i, j, k) * PMLcoef(BsigmaX(i, j, k)) +
                    PMLcoef2_x * (Ez(i_next, j, k) - Ez(i, j, k));
        Bzx(i, j, k) = Bzx(i, j, k) * PMLcoef(BsigmaX(i, j, k)) -
                    PMLcoef2_x * (Ey(i_next, j, k) - Ey(i, j, k));

        Bxy(i, j, k) = Bxy(i, j, k) * PMLcoef(BsigmaY(i, j, k)) -
                    PMLcoef2_y * (Ez(i, j_next, k) - Ez(i, j, k));
        Bzy(i, j, k) = Bzy(i, j, k) * PMLcoef(BsigmaY(i, j, k)) +
                    PMLcoef2_y * (Ex(i, j_next, k) - Ex(i, j, k));

        Bxz(i, j, k) = Bxz(i, j, k) * PMLcoef(BsigmaZ(i, j, k)) +
                    PMLcoef2_z * (Ey(i, j, k_next) - Ey(i, j, k));
        Byz(i, j, k) = Byz(i, j, k) * PMLcoef(BsigmaZ(i, j, k)) -
                    PMLcoef2_z * (Ex(i, j, k_next) - Ex(i, j, k));

        Bx(i, j, k) = Bxy(i, j, k) + Bxz(i, j, k);
        By(i, j, k) = Byz(i, j, k) + Byx(i, j, k);
        Bz(i, j, k) = Bzx(i, j, k) + Bzy(i, j, k);
    }
};


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
    int pml_size_i, pml_size_j, pml_size_k;
    std::function<double(double, double, double, double)> init_func;

    int time;

    void write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it);

public:
    FDTD(Parameters _parameters, CurrentParameters _Cpar, double _dt, double _pml_percent,
    std::function<double(double, double, double, double)> init_function, int time_);

    Field& get_field(Component);

    void update_fields(bool write_result = false, 
        Axis write_axis = Axis::X, std::string base_path = "");
};


FDTD::FDTD(Parameters _parameters, CurrentParameters _Cpar, double _dt, double _pml_percent,
    std::function<double(double, double, double, double)> init_function, int time_) :
    parameters(_parameters), cParams(_Cpar), dt(_dt), pml_percent(_pml_percent), init_func(init_function), time(time_)

{
    if (parameters.Ni <= 0 ||
        parameters.Nj <= 0 ||
        parameters.Nk <= 0 ||
        dt <= 0)
    {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    Jx = TimeField("Jx", time, parameters.Ni, parameters.Nj, parameters.Nk);
    Jy = TimeField("Jy", time, parameters.Ni, parameters.Nj, parameters.Nk);
    Jz = TimeField("Jz", time, parameters.Ni, parameters.Nj, parameters.Nk);

    Ex = Field("Ex", parameters.Ni, parameters.Nj, parameters.Nk);
    Ey = Field("Ey", parameters.Ni, parameters.Nj, parameters.Nk);
    Ez = Field("Ez", parameters.Ni, parameters.Nj, parameters.Nk);
    Bx = Field("Bx", parameters.Ni, parameters.Nj, parameters.Nk);
    By = Field("By", parameters.Ni, parameters.Nj, parameters.Nk);
    Bz = Field("Bz", parameters.Ni, parameters.Nj, parameters.Nk);

    Exy = Field("Exy", parameters.Ni, parameters.Nj, parameters.Nk);
    Exz = Field("Exz", parameters.Ni, parameters.Nj, parameters.Nk);
    Eyx = Field("Eyx", parameters.Ni, parameters.Nj, parameters.Nk);
    Eyz = Field("Eyz", parameters.Ni, parameters.Nj, parameters.Nk);
    Ezx = Field("Ezx", parameters.Ni, parameters.Nj, parameters.Nk);
    Ezy = Field("Ezy", parameters.Ni, parameters.Nj, parameters.Nk);

    Bxy = Field("Bxy", parameters.Ni, parameters.Nj, parameters.Nk);
    Bxz = Field("Bxz", parameters.Ni, parameters.Nj, parameters.Nk);
    Byx = Field("Byx", parameters.Ni, parameters.Nj, parameters.Nk);
    Byz = Field("Byz", parameters.Ni, parameters.Nj, parameters.Nk);
    Bzx = Field("Bzx", parameters.Ni, parameters.Nj, parameters.Nk);
    Bzy = Field("Bzy", parameters.Ni, parameters.Nj, parameters.Nk);

    EsigmaX = Field("EsigmaX", parameters.Ni, parameters.Nj, parameters.Nk);
    EsigmaY = Field("EsigmaY", parameters.Ni, parameters.Nj, parameters.Nk);
    EsigmaZ = Field("EsigmaZ", parameters.Ni, parameters.Nj, parameters.Nk);

    BsigmaX = Field("BsigmaX", parameters.Ni, parameters.Nj, parameters.Nk);
    BsigmaY = Field("BsigmaY", parameters.Ni, parameters.Nj, parameters.Nk);
    BsigmaZ = Field("BsigmaZ", parameters.Ni, parameters.Nj, parameters.Nk);

    pml_size_i = static_cast<int>(static_cast<double>(parameters.Ni) * pml_percent);
    pml_size_j = static_cast<int>(static_cast<double>(parameters.Nj) * pml_percent);
    pml_size_k = static_cast<int>(static_cast<double>(parameters.Nk) * pml_percent);
}

Field& FDTD::get_field(Component this_field)
{
    switch (this_field)
    {
    case Component::EX: return Ex;

    case Component::EY: return Ey;

    case Component::EZ: return Ez;

    case Component::BX: return Bx;

    case Component::BY: return By;

    case Component::BZ: return Bz;

    default: throw std::logic_error("ERROR: Invalid field component");
    }
}

class InitializeCurrentFunctor {
public:
    
    InitializeCurrentFunctor(TimeField J, CurrentParameters cParams, Parameters parameters,
    std::function<double(double, double, double, double)> init_function)
        : J(J), cParams(cParams), parameters(parameters), init_function(init_function) {}

    static void apply(TimeField J, CurrentParameters cParams, Parameters parameters,
                std::function<double(double, double, double, double)> init_function,
                int iterations, int bounds_i[2], int bounds_j[2], int bounds_k[2]) {
        
        InitializeCurrentFunctor functor(J, cParams, parameters, init_function);

        Kokkos::MDRangePolicy<Kokkos::Rank<4>> policy({0, bounds_i[0], bounds_j[0], bounds_k[0]},
                                                      {iterations, bounds_i[1], bounds_j[1], bounds_k[1]});

        Kokkos::parallel_for("UpdateBPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int iter, const int i, const int j, const int k) const {

        J(iter, i, j, k) = init_function(static_cast<double>(i) * parameters.dx,
                                        static_cast<double>(j) * parameters.dy,
                                        static_cast<double>(k) * parameters.dz,
                                        static_cast<double>(iter + 1) * cParams.dt);
    }

private:
    TimeField J;
    CurrentParameters cParams;
    Parameters parameters;
    std::function<double(double, double, double, double)> init_function;
};

void FDTD::update_fields(bool write_result, Axis write_axis, std::string base_path)
{
    if (time < 0)
    {
        throw std::invalid_argument("ERROR: Invalid update field argument");
    }

    double Tx = cParams.period_x;
    double Ty = cParams.period_y;
    double Tz = cParams.period_z;
        
    int start_i = static_cast<int>(floor((-Tx / 4.0 - parameters.ax) / parameters.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - parameters.ay) / parameters.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - parameters.az) / parameters.dz));

    int max_i = static_cast<int>(floor((Tx / 4.0 - parameters.ax) / parameters.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - parameters.ay) / parameters.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - parameters.az) / parameters.dz));

    int size_i_cur[2] = { start_i, max_i };
    int size_j_cur[2] = { start_j, max_j };
    int size_k_cur[2] = { start_k, max_k };

    InitializeCurrentFunctor::apply(Jx, cParams, parameters, init_func, cParams.iterations,
                                    size_i_cur, size_j_cur, size_k_cur);
    InitializeCurrentFunctor::apply(Jy, cParams, parameters, init_func, cParams.iterations,
                                    size_i_cur, size_j_cur, size_k_cur);
    InitializeCurrentFunctor::apply(Jz, cParams, parameters, init_func, cParams.iterations,
                                    size_i_cur, size_j_cur, size_k_cur);

    if (pml_percent == 0.0)
    {
        int size_i_main[2] = { 0, parameters.Ni };
        int size_j_main[2] = { 0, parameters.Nj };
        int size_k_main[2] = { 0, parameters.Nk };
        for (int t = 0; t < time; t++)
        {
            std::cout << "Iteration: " << t << std::endl;

            ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt, 
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main,
            parameters.Ni, parameters.Nj, parameters.Nk);

            ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
             Jx, Jy, Jz, dt, 
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main, t, cParams.iterations,
            parameters.Ni, parameters.Nj, parameters.Nk);

            ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main,
            parameters.Ni, parameters.Nj, parameters.Nk);

            auto Ex_host = Kokkos::create_mirror_view(Ex);
            auto Ey_host = Kokkos::create_mirror_view(Ey);
            auto Ez_host = Kokkos::create_mirror_view(Ez);
            auto Bx_host = Kokkos::create_mirror_view(Bx);
            auto By_host = Kokkos::create_mirror_view(By);
            auto Bz_host = Kokkos::create_mirror_view(Bz);

            Kokkos::deep_copy(Ex_host, Ex);
            Kokkos::deep_copy(Ey_host, Ey);
            Kokkos::deep_copy(Ez_host, Ez);
            Kokkos::deep_copy(Bx_host, Bx);
            Kokkos::deep_copy(By_host, By);
            Kokkos::deep_copy(Bz_host, Bz);

            std::vector<Field> return_data{ Ex_host, Ey_host, Ez_host, Bx_host, By_host, Bz_host };
            if (write_result)
                write_spherical(return_data, write_axis, base_path, t);
        }
        return;
    }

    // // Defining areas of computation
    // int size_i_main[] = { pml_size_i, parameters.Ni - pml_size_i };
    // int size_j_main[] = { pml_size_j, parameters.Nj - pml_size_j };
    // int size_k_main[] = { pml_size_k, parameters.Nk - pml_size_k };

    // int size_i_solid[] = { 0, parameters.Ni };
    // int size_j_solid[] = { 0, parameters.Nj };
    // int size_k_solid[] = { 0, parameters.Nk };

    // int size_i_part_from_start[] = { 0, parameters.Ni - pml_size_i };
    // int size_i_part_from_end[] = { pml_size_i, parameters.Ni };

    // int size_k_part_from_start[] = { 0, parameters.Nk - pml_size_k };
    // int size_k_part_from_end[] = { pml_size_k, parameters.Nk };

    // int size_xy_lower_k_pml[] = { 0, pml_size_k };
    // int size_xy_upper_k_pml[] = { parameters.Nk - pml_size_k, parameters.Nk };

    // int size_yz_lower_i_pml[] = { 0, pml_size_i };
    // int size_yz_upper_i_pml[] = { parameters.Ni - pml_size_i, parameters.Ni };

    // int size_zx_lower_j_pml[] = { 0, pml_size_j };
    // int size_zx_upper_j_pml[] = { parameters.Nj - pml_size_j, parameters.Nj };

    // // Definition of functions for calculating the distance to the interface
    // std::function<int(int, int, int)> calc_distant_i_up =
    //     [=](int i, int j, int k) {
    //     return i + 1 + pml_size_i - parameters.Ni;
    // };
    // std::function<int(int, int, int)> calc_distant_j_up =
    //     [=](int i, int j, int k) {
    //     return j + 1 + pml_size_j - parameters.Nj;
    // };
    // std::function<int(int, int, int)> calc_distant_k_up =
    //     [=](int i, int j, int k) {
    //     return k + 1 + pml_size_k - parameters.Nk;
    // };

    // std::function<int(int, int, int)> calc_distant_i_low =
    //     [=](int i, int j, int k) {
    //     return pml_size_i - i;
    // };
    // std::function<int(int, int, int)> calc_distant_j_low =
    //     [=](int i, int j, int k) {
    //     return pml_size_j - j;
    // };
    // std::function<int(int, int, int)> calc_distant_k_low =
    //     [=](int i, int j, int k) {
    //     return pml_size_k - k;
    // };

    // // Calculation of maximum permittivity and permeability
    // double SGm_x = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
    //     / (static_cast<double>(pml_size_i) * parameters.dx);
    // double SGm_y = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
    //     / (static_cast<double>(pml_size_j) * parameters.dy);
    // double SGm_z = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
    //     / (static_cast<double>(pml_size_k) * parameters.dz);

    // // Calculation of permittivity and permeability in the cells
    // set_sigma_z(size_i_solid, size_j_solid, size_xy_lower_k_pml,
    //     SGm_z, calc_distant_k_low);
    // set_sigma_y(size_i_solid, size_zx_lower_j_pml, size_k_solid,
    //     SGm_y, calc_distant_j_low);
    // set_sigma_x(size_yz_lower_i_pml, size_j_solid, size_k_solid,
    //     SGm_x, calc_distant_i_low);

    // set_sigma_z(size_i_solid, size_j_solid, size_xy_upper_k_pml,
    //     SGm_z, calc_distant_k_up);
    // set_sigma_y(size_i_solid, size_zx_upper_j_pml, size_k_solid,
    //     SGm_y, calc_distant_j_up);
    // set_sigma_x(size_yz_upper_i_pml, size_j_solid, size_k_solid,
    //     SGm_x, calc_distant_i_up);

    // for (int t = 0; t < time; t++)
    // {
    //     std::cout << "Iteration: " << t + 1 << std::endl;

    //     update_B(size_i_main, size_j_main, size_k_main);

    //     update_B_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
    //     update_B_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
    //     update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

    //     update_B_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
    //     update_B_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
    //     update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

    //     update_E(size_i_main, size_j_main, size_k_main, t);

    //     update_E_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
    //     update_E_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
    //     update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

    //     update_E_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
    //     update_E_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
    //     update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

    //     update_B(size_i_main, size_j_main, size_k_main);

    //     std::vector<Field> new_iteration{ Ex, Ey, Ez, Bx, By, Bz };
    //     return_data = new_iteration;
    //     if (write_result)
    //         write_spherical(return_data, write_axis, base_path, t);
    // }
}


void FDTD::write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it)
{
    std::ofstream test_fout;
    switch (axis)
    {
    case Axis::X:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int k = 0; k < parameters.Nk; ++k)
            {
                for (int j = 0; j < parameters.Nj; ++j)
                {
                    test_fout << field(parameters.Ni / 2, j, k);
                    if (j == parameters.Nj - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    case Axis::Y:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int i = 0; i < parameters.Nj; ++i)
            {
                for (int k = 0; k < parameters.Nk; ++k)
                {
                    test_fout << field(i, parameters.Nj / 2, k);
                    if (k == parameters.Nk - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    case Axis::Z:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int i = 0; i < parameters.Nj; ++i)
            {
                for (int j = 0; j < parameters.Nk; ++j)
                {
                    test_fout << field(i, j, parameters.Nk / 2);
                    if (j == parameters.Nk - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    default: throw std::logic_error("ERROR: Invalid axis");
    }
}