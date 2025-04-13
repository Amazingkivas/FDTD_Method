#pragma once

#include "kokkos_shared.h"
#include "Structures.h"
#include "FDTD_boundaries.h"

#include <chrono>
#include <Kokkos_SIMD.hpp>

using namespace FDTD_struct;

namespace FDTD_kokkos
{

class ComputeSigmaFunctor
{
private:
    double SGm, dt;
    Function dist;
    int pml_size;
    Field Esigma, Bsigma;
public:
    ComputeSigmaFunctor(Field _Esigma, Field _Bsigma, double _SGm, Function distance, int _pml_size, double _dt) :
       Esigma(_Esigma), Bsigma(_Bsigma), SGm(_SGm), dist(distance), pml_size(_pml_size), dt(_dt) {}

    static void apply(Field _Esigma, Field _Bsigma, double _SGm, Function distance, int _pml_size, double dt,
       int bounds_i[2], int bounds_j[2], int bounds_k[2])
    {
        ComputeSigmaFunctor functor(_Esigma, _Bsigma, _SGm, distance, _pml_size, dt);
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            { bounds_i[0], bounds_j[0], bounds_k[0] }, { bounds_i[1], bounds_j[1], bounds_k[1] }),
            functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& i, const int& j, const int& k) const
    {
        /*Esigma(i, j, k) = SGm * std::pow(static_cast<double>(dist(i, j, k))
           / static_cast<double>(pml_size), FDTD_const::N);
        Bsigma(i, j, k) = SGm * std::pow(static_cast<double>(dist(i, j, k))
           / static_cast<double>(pml_size), FDTD_const::N);*/
    }
};

class ComputeE_FieldFunctor
{
private:
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    Field &Jx, &Jy, &Jz;
    int t, iters;
    int Ni, Nj, Nk;
    double current_coef;
    double coef_dx, coef_dy, coef_dz;
    KOKKOS_INLINE_FUNCTION
    void applyPeriodicBoundary(int& i, int& j, int& k) const
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
public:
    ComputeE_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bx, Field& By, Field& Bz,
                        Field& Jx, Field& Jy, Field& Jz,
                        double& current_coef, int& t, int& iters,
                        int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz),
          Jx(Jx), Jy(Jy), Jz(Jz), current_coef(current_coef), t(t), iters(iters), 
          Ni(Ni), Nj(Nj), Nk(Nk), coef_dx(coef_dx), coef_dy(coef_dy), coef_dz(coef_dz) {}

    static void apply(Field& Ex, Field& Ey, Field& Ez,
                      Field& Bx, Field& By, Field& Bz,
                      Field& Jx, Field& Jy, Field& Jz,
                      double& current_coef,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2], int& t, int& iters,
                      int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz) {
        
        ComputeE_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, current_coef, t, iters, Ni, Nj, Nk, coef_dx, coef_dy, coef_dz);

        //int begin = bounds_i[0] + 1 + (bounds_j[0] + 1) * Ni + (bounds_k[0] + 1) * Ni * Nj;
        //int end = bounds_i[1] + bounds_j[1] * Ni + bounds_k[1] * Ni * Nj;

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_k[0] + 1, bounds_j[0] + 1, bounds_i[0] + 1},
                                                      {bounds_k[1], bounds_j[1], bounds_i[1]});

        Kokkos::parallel_for("UpdateEField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& k, const int& j, const int& i) const 
    {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred);

        int index = i + j * Ni + k * Nj * Ni;

        i_pred = i_pred + j * Ni + k * Ni * Nj;
        j_pred = i + j_pred * Ni + k * Ni * Nj;
        k_pred = i + j * Ni + k_pred * Ni * Nj;

	Ex[index] += -current_coef * Jx[index] + coef_dy * (Bz[index] - Bz[j_pred]) -
                               coef_dz * (By[index] - By[k_pred]);
        Ey[index] += -current_coef * Jy[index] + coef_dz * (Bx[index] - Bx[k_pred]) -
                               coef_dx * (Bz[index] - Bz[i_pred]);
        Ez[index] += -current_coef * Jz[index] + coef_dx * (By[index] - By[i_pred]) -
                               coef_dy * (Bx[index] - Bx[j_pred]);
    }
};

class ComputeB_FieldFunctor
{
private:
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    int Ni, Nj, Nk;
    double coef_dx, coef_dy, coef_dz;
    KOKKOS_INLINE_FUNCTION
    void applyPeriodicBoundary(int& i, int& j, int& k) const
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
public:
    ComputeB_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bx, Field& By, Field& Bz,
                        int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz), Ni(Ni), Nj(Nj), Nk(Nk),
          coef_dx(coef_dx), coef_dy(coef_dy), coef_dz(coef_dz) {}

    static void apply(Field& Ex, Field& Ey, Field& Ez,
                      Field& Bx, Field& By, Field& Bz,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2],
                      int Ni, int Nj, int Nk, double coef_dx, double coef_dy, double coef_dz) {
        
        ComputeB_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, Ni, Nj, Nk, coef_dx, coef_dy, coef_dz);

        //int begin = bounds_i[0] + 1 + (bounds_j[0] + 1) * Ni + (bounds_k[0] + 1) * Ni * Nj;
        //int end = bounds_i[1] + bounds_j[1] * Ni + bounds_k[1] * Ni * Nj;

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_k[0] + 1, bounds_j[0] + 1, bounds_i[0] + 1},
                                                      {bounds_k[1], bounds_j[1], bounds_i[1]});

        Kokkos::parallel_for("UpdateBField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& k, const int& j, const int& i) const
    {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next);

        int index = i + j * Ni + k * Nj * Ni;

        i_next = i_next + j * Ni + k * Ni * Nj;
        j_next = i + j_next * Ni + k * Ni * Nj;
        k_next = i + j * Ni + k_next * Ni * Nj;

        Bx[index] += coef_dz * (Ey[k_next] - Ey[index]) -
                     coef_dy * (Ez[j_next] - Ez[index]);
        By[index] += coef_dx * (Ez[i_next] - Ez[index]) -
                     coef_dz * (Ex[k_next] - Ex[index]);
        Bz[index] += coef_dy * (Ex[j_next] - Ex[index]) -
                     coef_dx * (Ey[i_next] - Ey[index]);
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
        return std::exp(-sigma * dt * FDTD_const::C);
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
        /*int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        if (EsigmaX(i, j, k) != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(EsigmaX(i, j, k))) / (EsigmaX(i, j, k) * dx);
        else
            PMLcoef2_x = FDTD_const::C * dt / dx;

        if (EsigmaY(i, j, k) != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(EsigmaY(i, j, k))) / (EsigmaY(i, j, k) * dy);
        else
            PMLcoef2_y = FDTD_const::C * dt / dy;

        if (EsigmaZ(i, j, k) != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ(i, j, k))) / (EsigmaZ(i, j, k) * dz);
        else
            PMLcoef2_z = FDTD_const::C * dt / dz;

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
        Ez(i, j, k) = Ezy(i, j, k) + Ezx(i, j, k);*/
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
        return std::exp(-sigma * dt * FDTD_const::C);
    }
public:
    ComputeB_PML_FieldFunctor(Field Ex, Field Ey, Field Ez,
                      Field Bxy, Field Byx, Field Bzy,
                      Field Byz, Field Bzx, Field Bxz,
                      Field Bx, Field By, Field Bz,
                      Field BsigmaX, Field BsigmaY, Field BsigmaZ,
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
        /*int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        if (BsigmaX(i, j, k) != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(BsigmaX(i, j, k))) / (BsigmaX(i, j, k) * dx);
        else
            PMLcoef2_x = FDTD_const::C * dt / dx;

        if (BsigmaY(i, j, k) != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(BsigmaY(i, j, k))) / (BsigmaY(i, j, k) * dy);
        else
            PMLcoef2_y = FDTD_const::C * dt / dy;

        if (BsigmaZ(i, j, k) != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ(i, j, k))) / (BsigmaZ(i, j, k) * dz);
        else
            PMLcoef2_z = FDTD_const::C * dt / dz;

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
        Bz(i, j, k) = Bzx(i, j, k) + Bzy(i, j, k);*/
    }
};

}

