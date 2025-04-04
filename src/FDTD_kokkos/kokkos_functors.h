#pragma once

#include "kokkos_shared.h"
#include "Structures.h"
#include "FDTD_boundaries.h"

#include <chrono>
#include <Kokkos_SIMD.hpp>

using namespace FDTD_struct;

namespace FDTD_kokkos {

class ComputeSigmaFunctor {
private:
    double SGm, dt;
    Function dist;
    int pml_size, Ni, Nj, Nk;
    Field Esigma, Bsigma;
public:
    ComputeSigmaFunctor(Field _Esigma, Field _Bsigma, double _SGm,
        Function distance, int _pml_size, double _dt, int _Ni, int _Nj, int _Nk) :
        Esigma(_Esigma), Bsigma(_Bsigma), SGm(_SGm), dist(distance),
        pml_size(_pml_size), dt(_dt), Ni(_Ni), Nj(_Nj), Ni(_Nj) {}

    static void apply(Field _Esigma, Field _Bsigma, double _SGm, Function distance,
        int _pml_size, double dt, Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
        ComputeSigmaFunctor functor(_Esigma, _Bsigma, _SGm, distance, _pml_size, dt);
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            { bounds_i.first, bounds_j.first, bounds_k.first },
            { bounds_i.second, bounds_j.second, bounds_k.second }), functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& i, const int& j, const int& k) const {
        int index = i + j * Ni + k * Ni * Nj;

        Esigma[index] = SGm * std::pow(static_cast<double>(dist[index])
           / static_cast<double>(pml_size), FDTD_const::N);
        Bsigma[index] = SGm * std::pow(static_cast<double>(dist[index])
           / static_cast<double>(pml_size), FDTD_const::N);
    }
};


class ComputeE_PML_FieldFunctor {
private:
    Field Ex, Ey, Ez;
    Field Exy, Eyx, Ezy, Eyz, Ezx, Exz;
    Field Bx, By, Bz;
    Field EsigmaX, EsigmaY, EsigmaZ;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;

    double PMLcoef(double sigma) const {
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
                      Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
                      int Ni, int Nj, int Nk) {
        
        ComputeE_PML_FieldFunctor functor(Ex, Ey, Ez, Exy, Eyx, Ezy, Eyz, Ezx, Exz,
                                   Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
                                   dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_i.first, bounds_j.first, bounds_k.first},
                                                      {bounds_i.second, bounds_j.second, bounds_k.second});

        Kokkos::parallel_for("UpdateEPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& i, const int& j, const int& k) const {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

        i_pred = i_pred + j * Ni + k * Ni * Nj;
        j_pred = i + j_pred * Ni + k * Ni * Nj;
        k_pred = i + j * Ni + k_pred * Ni * Nj;

        int index = i + j * Ni + k * Ni * Nj;

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        if (EsigmaX[index] != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(EsigmaX[index])) / (EsigmaX[index] * dx);
        else
            PMLcoef2_x = FDTD_const::C * dt / dx;

        if (EsigmaY[index] != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(EsigmaY[index])) / (EsigmaY[index] * dy);
        else
            PMLcoef2_y = FDTD_const::C * dt / dy;

        if (EsigmaZ[index] != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ[index])) / (EsigmaZ[index] * dz);
        else
            PMLcoef2_z = FDTD_const::C * dt / dz;

        Eyx[index] = Eyx[index] * PMLcoef(EsigmaX[index]) -
                    PMLcoef2_x * (Bz[index] - Bz[i_pred]);
        Ezx[index] = Ezx[index] * PMLcoef(EsigmaX[index]) +
                    PMLcoef2_x * (By[index] - By[i_pred]);

        Exy[index] = Exy[index] * PMLcoef(EsigmaY[index]) +
                    PMLcoef2_y * (Bz[index] - Bz[j_pred]);
        Ezy[index] = Ezy[index] * PMLcoef(EsigmaY[index]) -
                    PMLcoef2_y * (Bx[index] - Bx[j_pred]);

        Exz[index] = Exz[index] * PMLcoef(EsigmaZ[index]) -
                    PMLcoef2_z * (By[index] - By[k_pred]);
        Eyz[index] = Eyz[index] * PMLcoef(EsigmaZ[index]) +
                    PMLcoef2_z * (Bx[index] - Bx[k_pred]);

        Ex[index] = Exz[index] + Exy[index];
        Ey[index] = Eyx[index] + Eyz[index];
        Ez[index] = Ezy[index] + Ezx[index];
    }
};

class ComputeB_PML_FieldFunctor {
private:
    Field Ex, Ey, Ez;
    Field Bxy, Byx, Bzy, Byz, Bzx, Bxz;
    Field Bx, By, Bz;
    Field BsigmaX, BsigmaY, BsigmaZ;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;

    double PMLcoef(double sigma) const {
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
                      Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
                      int Ni, int Nj, int Nk) {

        ComputeB_PML_FieldFunctor functor(Ex, Ey, Ez, Bxy, Byx, Bzy, Byz, Bzx, Bxz,
                                   Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
                                   dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_i.first, bounds_j.first, bounds_k.first},
                                                      {bounds_i.second, bounds_j.second, bounds_k.second});

        Kokkos::parallel_for("UpdateBPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& i, const int& j, const int& k) const {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

        i_next = i_next + j * Ni + k * Ni * Nj;
        j_next = i + j_next * Ni + k * Ni * Nj;
        k_next = i + j * Ni + k_next * Ni * Nj;

        int index = i + j * Ni + k * Ni * Nj;

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        if (BsigmaX[index] != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(BsigmaX[index])) / (BsigmaX[index] * dx);
        else
            PMLcoef2_x = FDTD_const::C * dt / dx;

        if (BsigmaY[index] != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(BsigmaY[index])) / (BsigmaY[index] * dy);
        else
            PMLcoef2_y = FDTD_const::C * dt / dy;

        if (BsigmaZ[index] != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ[index])) / (BsigmaZ[index] * dz);
        else
            PMLcoef2_z = FDTD_const::C * dt / dz;

        Byx[index] = Byx[index] * PMLcoef(BsigmaX[index]) +
            PMLcoef2_x * (Ez[i_next] - Ez[index]);
        Bzx[index] = Bzx[index] * PMLcoef(BsigmaX[index]) -
            PMLcoef2_x * (Ey[i_next] - Ey[index]);

        Bxy[index] = Bxy[index] * PMLcoef(BsigmaY[index]) -
            PMLcoef2_y * (Ez[j_next] - Ez[index]);
        Bzy[index] = Bzy[index] * PMLcoef(BsigmaY[index]) +
            PMLcoef2_y * (Ex[j_next] - Ex[index]);

        Bxz[index] = Bxz[index] * PMLcoef(BsigmaZ[index]) +
            PMLcoef2_z * (Ey[k_next] - Ey[index]);
        Byz[index] = Byz[index] * PMLcoef(BsigmaZ[index]) -
            PMLcoef2_z * (Ex[k_next] - Ex[index]);

        Bx[index] = Bxy[index] + Bxz[index];
        By[index] = Byz[index] + Byx[index];
        Bz[index] = Bzx[index] + Bzy[index];
    }
};

}
