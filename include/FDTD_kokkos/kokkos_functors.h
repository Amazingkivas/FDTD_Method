#pragma once

#include "kokkos_shared.h"


namespace FDTD_kokkos {

using Boundaries = std::pair<int, int>;

class ComputeE_FieldFunctor {
private:
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    Field &Jx, &Jy, &Jz;
    int start_i, end_i;
    int Ni, Nj, Nk;
    FP current_coef;
    FP dx, dy, dz, dt;

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
    ComputeE_FieldFunctor(
        Field& Ex, Field& Ey, Field& Ez,
        Field& Bx, Field& By, Field& Bz,
        Field& Jx, Field& Jy, Field& Jz,
        const FP& current_coef,
        const int& start_i, const int& end_i,
        const int& Ni, const int& Nj, const int& Nk,
        const FP& dx, const FP& dy, const FP& dz, const FP& dt) :
        Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz),
        Jx(Jx), Jy(Jy), Jz(Jz), current_coef(current_coef),
        start_i(start_i), end_i(end_i),
        Ni(Ni), Nj(Nj), Nk(Nk),
        dx(dx), dy(dy), dz(dz), dt(dt) {}

    static void apply(
        Field& Ex, Field& Ey, Field& Ez,
        Field& Bx, Field& By, Field& Bz,
        Field& Jx, Field& Jy, Field& Jz,
        const FP& current_coef,
        const int bounds_i[2], const int bounds_j[2], const int bounds_k[2],
        const int& Ni, const int& Nj, const int& Nk,
        const FP& dx, const FP& dy, const FP& dz, const FP& dt) {
        
        ComputeE_FieldFunctor functor(
            Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz,
            current_coef, bounds_i[0], bounds_i[1],
            Ni, Nj, Nk, dx, dy, dz, dt);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_k[0], bounds_j[0], bounds_i[0]},
                                                      {bounds_k[1], bounds_j[1], bounds_i[1]});

        Kokkos::parallel_for("UpdateEField", policy, functor);
    }
    KOKKOS_INLINE_FUNCTION void operator()(const int& k, const int& j, const int& i) const {

        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred);

        int index = i + j * Ni + k * Nj * Ni;

        i_pred = i_pred + j * Ni + k * Ni * Nj;
        j_pred = i + j_pred * Ni + k * Ni * Nj;
        k_pred = i + j * Ni + k_pred * Ni * Nj;

	    Ex[index] += current_coef * Jx[index] + FDTD_const::C * dt * ((Bz[index] - Bz[j_pred]) / dy -
        (By[index] - By[k_pred]) / dz);
        Ey[index] += current_coef * Jy[index] + FDTD_const::C * dt * ((Bx[index] - Bx[k_pred]) / dz -
        (Bz[index] - Bz[i_pred]) / dx);
        Ez[index] += current_coef * Jz[index] + FDTD_const::C * dt * ((By[index] - By[i_pred]) / dx -
        (Bx[index] - Bx[j_pred]) / dy);
}
};

class ComputeB_FieldFunctor {
private:
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    int Ni, Nj, Nk;
    FP dx, dy, dz, dt;
    int start_i, end_i;

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
    ComputeB_FieldFunctor(
        Field& Ex, Field& Ey, Field& Ez,
        Field& Bx, Field& By, Field& Bz,
        const int& start_i, const int& end_i,
        const int& Ni, const int& Nj, const int& Nk,
        const FP& dx, const FP& dy, const FP& dz, const FP& dt) :
        Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz),
        Ni(Ni), Nj(Nj), Nk(Nk), start_i(start_i), end_i(end_i),
        dx(dx), dy(dy), dz(dz) {}

    static void apply(
        Field& Ex, Field& Ey, Field& Ez,
        Field& Bx, Field& By, Field& Bz,
        const int bounds_i[2], const int bounds_j[2], const int bounds_k[2],
        const int& Ni, const int& Nj, const int& Nk,
        const FP& dx, const FP& dy, const FP& dz, const FP& dt) {
        
        ComputeB_FieldFunctor functor(
            Ex, Ey, Ez, Bx, By, Bz,
            bounds_i[0], bounds_i[1],
            Ni, Nj, Nk, dx, dy, dz, dt);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({bounds_k[0], bounds_j[0], bounds_i[0]},
                                                      {bounds_k[1], bounds_j[1], bounds_i[1]});

        Kokkos::parallel_for("UpdateBField", policy, functor);
    }
    KOKKOS_INLINE_FUNCTION void operator()(const int& k, const int& j, const int& i) const {
    
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next);

        int index = i + j * Ni + k * Nj * Ni;

        i_next = i_next + j * Ni + k * Ni * Nj;
        j_next = i + j_next * Ni + k * Ni * Nj;
        k_next = i + j * Ni + k_next * Ni * Nj;

        Bx[index] += FDTD_const::C * dt / 2.0 * ((Ey[k_next] - Ey[index]) / dz -
                     (Ez[j_next] - Ez[index]) / dy);

        By[index] += FDTD_const::C * dt / 2.0 * ((Ez[i_next] - Ez[index]) / dx -
                     (Ex[k_next] - Ex[index]) / dz);

        Bz[index] += FDTD_const::C * dt / 2.0 * ((Ex[j_next] - Ex[index]) / dy -
                     (Ey[i_next] - Ey[index]) / dx);
}
};

class ComputeSigmaFunctor {
private:
    FP SGm, dt;
    Function dist;
    int pml_size, Ni, Nj, Nk;
    Field Esigma, Bsigma;
public:
    ComputeSigmaFunctor(Field _Esigma, Field _Bsigma, const FP& _SGm,
        Function distance, const int& _pml_size, const FP& _dt,
        const int& _Ni, const int& _Nj, const int& _Nk) :
        Esigma(_Esigma), Bsigma(_Bsigma), SGm(_SGm), dist(distance),
        pml_size(_pml_size), dt(_dt), Ni(_Ni), Nj(_Nj), Nk(_Nk) {}

    static void apply(Field _Esigma, Field _Bsigma, const FP& _SGm, Function distance,
        const int& _pml_size, const FP& dt,
        Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
        const int& _Ni, const int& _Nj, const int& _Nk) {
        ComputeSigmaFunctor functor(_Esigma, _Bsigma, _SGm, distance,
            _pml_size, dt, _Ni, _Nj, _Nk);
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            { bounds_k.first, bounds_j.first, bounds_i.first },
            { bounds_k.second, bounds_j.second, bounds_i.second }), functor);
    }

    KOKKOS_INLINE_FUNCTION void
    operator()(const int& k, const int& j, const int& i) const {
        const int index = i + j * Ni + k * Ni * Nj;

        Esigma[index] = SGm * std::pow(static_cast<FP>(dist(i, j, k))
            / static_cast<FP>(pml_size), FDTD_const::N);
        Bsigma[index] = SGm * std::pow(static_cast<FP>(dist(i, j, k))
            / static_cast<FP>(pml_size), FDTD_const::N);
    }
};

class Base_PML_functor {
protected:
    int Ni, Nj, Nk;
    FP dt, dx, dy, dz;

    KOKKOS_INLINE_FUNCTION 
    void applyPeriodicBoundary(int& i, int& j, int& k) const {
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

    FP PMLcoef(const FP& sigma) const {
        return std::exp(-sigma * dt * FDTD_const::C);
    }
public:
    Base_PML_functor(const FP& dt,
        const FP& dx, const FP& dy, const FP& dz,
        const int& Ni, const int& Nj, const int& Nk) :
        dt(dt), dx(dx), dy(dy), dz(dz), Ni(Ni), Nj(Nj), Nk(Nk) {}
};

class ComputeE_PML_FieldFunctor : public Base_PML_functor {
private:
    Field Ex, Ey, Ez;
    Field Exy, Eyx, Ezy, Eyz, Ezx, Exz;
    Field Bx, By, Bz;
    Field EsigmaX, EsigmaY, EsigmaZ;
    
public:
    ComputeE_PML_FieldFunctor(
        Field Ex, Field Ey, Field Ez,
        Field Exy, Field Eyx, Field Ezy,
        Field Eyz, Field Ezx, Field Exz,
        Field Bx, Field By, Field Bz,
        Field EsigmaX, Field EsigmaY, Field EsigmaZ,
        const FP& dt, 
        const FP& dx, const FP& dy, const FP& dz,
        const int& Ni, const int& Nj, const int& Nk
    ) :
        Ex(Ex), Ey(Ey), Ez(Ez), Exy(Exy), Eyx(Eyx), Ezy(Ezy),
        Eyz(Eyz), Ezx(Ezx), Exz(Exz), Bx(Bx), By(By), Bz(Bz),
        EsigmaX(EsigmaX), EsigmaY(EsigmaY), EsigmaZ(EsigmaZ),
        Base_PML_functor(dt, dx, dy, dz, Ni, Nj, Nk) {}

    static void apply(
        Field Ex, Field Ey, Field Ez,
        Field Exy, Field Eyx, Field Ezy,
        Field Eyz, Field Ezx, Field Exz,
        Field Bx, Field By, Field Bz,
        Field EsigmaX, Field EsigmaY, Field EsigmaZ,
        const FP& dt, 
        const FP& dx, const FP& dy, const FP& dz,
        Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
        const int& Ni, const int& Nj, const int& Nk) {

        ComputeE_PML_FieldFunctor functor(
            Ex, Ey, Ez, Exy, Eyx, Ezy, Eyz, Ezx, Exz,
            Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
            dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
            {bounds_k.first, bounds_j.first, bounds_i.first},
            {bounds_k.second, bounds_j.second, bounds_i.second});

        Kokkos::parallel_for("UpdateEPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& k, const int& j, const int& i) const {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(i_pred, j_pred, k_pred);

        i_pred = i_pred + j * Ni + k * Ni * Nj;
        j_pred = i + j_pred * Ni + k * Ni * Nj;
        k_pred = i + j * Ni + k_pred * Ni * Nj;

        const int index = i + j * Ni + k * Ni * Nj;

        FP PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

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

class ComputeB_PML_FieldFunctor : public Base_PML_functor {
private:
    Field Ex, Ey, Ez;
    Field Bxy, Byx, Bzy, Byz, Bzx, Bxz;
    Field Bx, By, Bz;
    Field BsigmaX, BsigmaY, BsigmaZ;

public:
    ComputeB_PML_FieldFunctor(
        Field Ex, Field Ey, Field Ez,
        Field Bxy, Field Byx, Field Bzy,
        Field Byz, Field Bzx, Field Bxz,
        Field Bx, Field By, Field Bz,
        Field BsigmaX, Field BsigmaY, Field BsigmaZ,
        const FP& dt, 
        const FP& dx, const FP& dy, const FP& dz,
        const int& Ni, const int& Nj, const int& Nk) : 
        Ex(Ex), Ey(Ey), Ez(Ez), Bxy(Bxy), Byx(Byx), Bzy(Bzy),
        Byz(Byz), Bzx(Bzx), Bxz(Bxz), Bx(Bx), By(By), Bz(Bz),
        BsigmaX(BsigmaX), BsigmaY(BsigmaY), BsigmaZ(BsigmaZ),
        Base_PML_functor(dt, dx, dy, dz, Ni, Nj, Nk) {}

    static void apply(
        Field Ex, Field Ey, Field Ez,
        Field Bxy, Field Byx, Field Bzy,
        Field Byz, Field Bzx, Field Bxz,
        Field Bx, Field By, Field Bz,
        Field BsigmaX, Field BsigmaY, Field BsigmaZ,
        const FP& dt, 
        const FP& dx, const FP& dy, const FP& dz,
        Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
        const int& Ni, const int& Nj, const int& Nk) {

        ComputeB_PML_FieldFunctor functor(
            Ex, Ey, Ez, Bxy, Byx, Bzy, Byz, Bzx, Bxz,
            Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
            dt, dx, dy, dz, Ni, Nj, Nk);

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
            {bounds_k.first, bounds_j.first, bounds_i.first},
            {bounds_k.second, bounds_j.second, bounds_i.second});

        Kokkos::parallel_for("UpdateBPMLField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& k, const int& j, const int& i) const {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(i_next, j_next, k_next);

        i_next = i_next + j * Ni + k * Ni * Nj;
        j_next = i + j_next * Ni + k * Ni * Nj;
        k_next = i + j * Ni + k_next * Ni * Nj;

        const int index = i + j * Ni + k * Ni * Nj;

        FP PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

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
