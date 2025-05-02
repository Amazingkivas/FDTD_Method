#pragma once

#include "kokkos_shared.h"
#include "Structures.h"
#include "FDTD_boundaries.h"
#include <numa.h>

#include <iostream>
#include <chrono>
#include <Kokkos_SIMD.hpp>

using namespace FDTD_struct;

namespace FDTD_kokkos
{

class ComputeE_FieldFunctor
{
private:
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    Field &Jx, &Jy, &Jz;
    int start_i, end_i;
    int Ni, Nj, Nk;
    double current_coef;
    double coef_dx, coef_dy, coef_dz;
    KOKKOS_INLINE_FUNCTION
    void applyPeriodicBoundary(int& i, const int& N) const
    {
        int i_isMinusOne = (i < 0);

        int i_isNi = (i == N);

        i = (N - 1) * i_isMinusOne + i *
            !(i_isMinusOne || i_isNi);
    }
public:
    ComputeE_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bx, Field& By, Field& Bz,
                        Field& Jx, Field& Jy, Field& Jz,
                        double& current_coef, int& start_i, int& end_i,
                        int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz),
          Jx(Jx), Jy(Jy), Jz(Jz), current_coef(current_coef), start_i(start_i), end_i(end_i), 
          Ni(Ni), Nj(Nj), Nk(Nk), coef_dx(coef_dx), coef_dy(coef_dy), coef_dz(coef_dz) {}

    static void apply(Field& Ex, Field& Ey, Field& Ez,
                      Field& Bx, Field& By, Field& Bz,
                      Field& Jx, Field& Jy, Field& Jz,
                      double& current_coef,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2], int& t, int& iters,
                      int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz) {
        
        ComputeE_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, current_coef, bounds_i[0], bounds_i[1], Ni, Nj, Nk, coef_dx, coef_dy, coef_dz);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({bounds_k[0], bounds_j[0]},
                                                      {bounds_k[1], bounds_j[1]});

        Kokkos::parallel_for("UpdateEField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& k, const int& j) const 
    {
        int j_pred = j - 1;
        int k_pred = k - 1;

        applyPeriodicBoundary(k_pred, Nk);
        applyPeriodicBoundary(j_pred, Nj);

        int index_kj = j * Ni + k * Ni * Nj;
        int j_pred_kj = j_pred * Ni + k * Ni * Nj;
        int k_pred_kj = j * Ni + k_pred * Ni * Nj;

        int i_base = start_i;
        for (i_base; i_base + simd_width <= end_i; i_base += simd_width) {
            simd_type Ex_simd, Ey_simd, Ez_simd;
            simd_type Bx_simd, By_simd, Bz_simd;
            simd_type Jx_simd, Jy_simd, Jz_simd;
            simd_type Bz_pred_simd, By_pred_simd, Bx_pred_simd;
            simd_type Bz_i_pred_simd, By_i_pred_simd, Bx_j_pred_simd;
            
            #pragma unroll
            for (int lane = 0; lane < simd_width; ++lane) {
                int i = i_base + lane;
                int index = i + index_kj;
                
                int i_pred = i - 1;
                
                applyPeriodicBoundary(i_pred, Ni);

                int i_pred_idx = i_pred + index_kj;
                int j_pred_idx = i + j_pred_kj;
                int k_pred_idx = i + k_pred_kj;
                
                Ex_simd[lane] = Ex[index];
                Ey_simd[lane] = Ey[index];
                Ez_simd[lane] = Ez[index];
                
                Bx_simd[lane] = Bx[index];
                By_simd[lane] = By[index];
                Bz_simd[lane] = Bz[index];

                Jx_simd[lane] = Jx[index];
                Jy_simd[lane] = Jy[index];
                Jz_simd[lane] = Jz[index];
                
                Bz_pred_simd[lane] = Bz[j_pred_idx];
                By_pred_simd[lane] = By[k_pred_idx];
                Bx_pred_simd[lane] = Bx[k_pred_idx];
                Bz_i_pred_simd[lane] = Bz[i_pred_idx];
                By_i_pred_simd[lane] = By[i_pred_idx];
                Bx_j_pred_simd[lane] = Bx[j_pred_idx];
            }
            
            Ex_simd += current_coef * Jx_simd + 
                coef_dy * (Bz_simd - Bz_pred_simd) -
                coef_dz * (By_simd - By_pred_simd);
                
            Ey_simd += current_coef * Jy_simd + 
                coef_dz * (Bx_simd - Bx_pred_simd) -
                coef_dx * (Bz_simd - Bz_i_pred_simd);
                
            Ez_simd += current_coef * Jz_simd + 
                coef_dx * (By_simd - By_i_pred_simd) -
                coef_dy * (Bx_simd - Bx_j_pred_simd);
            
            #pragma unroll
            for (int lane = 0; lane < simd_width; ++lane) {
                int i = i_base + lane;
                
                int index = i + index_kj;
                Ex[index] = Ex_simd[lane];
                Ey[index] = Ey_simd[lane];
                Ez[index] = Ez_simd[lane];
            }
        }
        for (int i = i_base; i < end_i; ++i) {
            const int index = i + index_kj;
            int i_pred = i - 1;
            std::cout << "k = " << k << " | j = " << j << " | i = " << i << std::endl;

            applyPeriodicBoundary(i_pred, Ni);
            
            const int i_pred_idx = i_pred + index_kj;
            const int j_pred_idx = i + j_pred_kj;
            const int k_pred_idx = i + k_pred_kj;

            Ex[index] += -current_coef * Jx[index] + coef_dy * (Bz[index] - Bz[j_pred_idx]) -
                               coef_dz * (By[index] - By[k_pred_idx]);
            Ey[index] += -current_coef * Jy[index] + coef_dz * (Bx[index] - Bx[k_pred_idx]) -
                               coef_dx * (Bz[index] - Bz[i_pred_idx]);
            Ez[index] += -current_coef * Jz[index] + coef_dx * (By[index] - By[i_pred_idx]) -
                               coef_dy * (Bx[index] - Bx[j_pred_idx]);
        }
    }
};

class ComputeB_FieldFunctor
{
private:
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    int Ni, Nj, Nk;
    double coef_dx, coef_dy, coef_dz;
    int start_i, end_i;
    KOKKOS_INLINE_FUNCTION
    void applyPeriodicBoundary(int& i, const int& N) const
    {
        int i_isMinusOne = (i < 0);

        int i_isNi = (i == N);

        i = (N - 1) * i_isMinusOne + i *
            !(i_isMinusOne || i_isNi);
    }
public:
    ComputeB_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bx, Field& By, Field& Bz, int& start_i, int& end_i,
                        int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz), Ni(Ni), Nj(Nj), Nk(Nk), start_i(start_i), end_i(end_i),
          coef_dx(coef_dx), coef_dy(coef_dy), coef_dz(coef_dz) {}

    static void apply(Field& Ex, Field& Ey, Field& Ez,
                      Field& Bx, Field& By, Field& Bz,
                      int bounds_i[2], int bounds_j[2], int bounds_k[2],
                      int Ni, int Nj, int Nk, double coef_dx, double coef_dy, double coef_dz) {
        
        ComputeB_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, bounds_i[0], bounds_i[1], Ni, Nj, Nk, coef_dx, coef_dy, coef_dz);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({bounds_k[0], bounds_j[0]},
                                                      {bounds_k[1], bounds_j[1]});

        Kokkos::parallel_for("UpdateBField", policy, functor);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int& k, const int& j) const
    {
        int j_next = j + 1;
        int k_next = k + 1;

        applyPeriodicBoundary(k_next, Nk);
        applyPeriodicBoundary(j_next, Nj);

        int index_kj = j * Ni + k * Ni * Nj;
        int j_next_kj = j_next * Ni + k * Ni * Nj;
        int k_next_kj = j * Ni + k_next * Ni * Nj;
        
        int i_base = start_i;
        for (i_base; i_base + simd_width <= end_i; i_base += simd_width) {
            simd_type Bx_simd, By_simd, Bz_simd;
            simd_type Ex_simd, Ey_simd, Ez_simd;
            simd_type Ey_next_simd, Ez_next_simd, Ex_next_simd;
            simd_type Ey_i_next_simd, Ez_j_next_simd, Ex_k_next_simd;

            #pragma unroll
            for (int lane = 0; lane < simd_width; ++lane) {
                int i = i_base + lane;
                
                int index = i + index_kj;
                
                int i_next = i + 1;
                
                applyPeriodicBoundary(i_next, Ni);
                
                int i_next_idx = i_next + index_kj;
                int j_next_idx = i + j_next_kj;
                int k_next_idx = i + k_next_kj;
                
                Bx_simd[lane] = Bx[index];
                By_simd[lane] = By[index];
                Bz_simd[lane] = Bz[index];
                
                Ex_simd[lane] = Ex[index];
                Ey_simd[lane] = Ey[index];
                Ez_simd[lane] = Ez[index];
                
                Ey_next_simd[lane] = Ey[k_next_idx];
                Ez_next_simd[lane] = Ez[i_next_idx];
                Ex_next_simd[lane] = Ex[j_next_idx];
                
                Ey_i_next_simd[lane] = Ey[i_next_idx];
                Ez_j_next_simd[lane] = Ez[j_next_idx];
                Ex_k_next_simd[lane] = Ex[k_next_idx];
            }
            
            Bx_simd += coef_dz * (Ey_next_simd - Ey_simd) -
                      coef_dy * (Ez_j_next_simd - Ez_simd);
                      
            By_simd += coef_dx * (Ez_next_simd - Ez_simd) -
                      coef_dz * (Ex_k_next_simd - Ex_simd);
                      
            Bz_simd += coef_dy * (Ex_next_simd - Ex_simd) -
                      coef_dx * (Ey_i_next_simd - Ey_simd);
            
            #pragma unroll
            for (int lane = 0; lane < simd_width; ++lane) {
                int i = i_base + lane;
                
                int index = i + index_kj;
                Bx[index] = Bx_simd[lane];
                By[index] = By_simd[lane];
                Bz[index] = Bz_simd[lane];
            }
        }
        for (int i = i_base; i < end_i; ++i) {
            const int index = i + index_kj;
            
            int i_next = i + 1;

            applyPeriodicBoundary(i_next, Ni);
            
            const int i_next_idx = i_next + index_kj;
            const int j_next_idx = i + j_next_kj;
            const int k_next_idx = i + k_next_kj;
            
            Bx[index] += coef_dz * (Ey[k_next_idx] - Ey[index]) -
                       coef_dy * (Ez[j_next_idx] - Ez[index]);
                       
            By[index] += coef_dx * (Ez[i_next_idx] - Ez[index]) -
                       coef_dz * (Ex[k_next_idx] - Ex[index]);
                       
            Bz[index] += coef_dy * (Ex[j_next_idx] - Ex[index]) -
                       coef_dx * (Ey[i_next_idx] - Ey[index]);
        }
    }
};

}

