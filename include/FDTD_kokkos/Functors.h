#pragma once

#include "kokkos_shared.h"
#include "Structures.h"
#include "FDTD_boundaries.h"

using namespace FDTDstruct;

namespace FDTD_kokkos
{

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

        Kokkos::parallel_for("InitializeCurrent", policy, functor);
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
        Esigma(i, j, k) = SGm * std::pow(static_cast<double>(dist(i, j, k))
           / static_cast<double>(pml_size), FDTDconst::N);
        Bsigma(i, j, k) = SGm * std::pow(static_cast<double>(dist(i, j, k))
           / static_cast<double>(pml_size), FDTDconst::N);
    }
};

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

        FDTD_boundaries::applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

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

        FDTD_boundaries::applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

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

        FDTD_boundaries::applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

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
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        FDTD_boundaries::applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

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

}
