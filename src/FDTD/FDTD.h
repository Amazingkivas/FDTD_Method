#pragma once

#include "shared.h"
#include "Structures.h"

namespace FDTD_openmp {

    using namespace FDTD_struct;

    class FDTD {
    private:
        Boundaries size_i_main, size_j_main, size_k_main;

    protected:
        Parameters parameters;

        Field Jx, Jy, Jz;
        Field Ex, Ey, Ez;
        Field Bx, By, Bz;

        double coef_J;
        double coef_E_dx, coef_E_dy, coef_E_dz;
        double coef_B_dx, coef_B_dy, coef_B_dz;

        int size;

        inline void applyPeriodicBoundary(int& i, int& j, int& k) {
            int i_isMinusOne = (i < 0);
            int j_isMinusOne = (j < 0);
            int k_isMinusOne = (k < 0);
        
            int i_isNi = (i == parameters.Ni);
            int j_isNj = (j == parameters.Nj);
            int k_isNk = (k == parameters.Nk);
        
            i = (parameters.Ni - 1) * i_isMinusOne + i *
                !(i_isMinusOne || i_isNi);
            j = (parameters.Nj - 1) * j_isMinusOne + j *
                !(j_isMinusOne || j_isNj);
            k = (parameters.Nk - 1) * k_isMinusOne + k *
                !(k_isMinusOne || k_isNk);
        }

        inline void update_E (Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
            int Ni = parameters.Ni;
            int Nj = parameters.Nj;
            int Nk = parameters.Nk;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; ++k) {
                for (int j = bounds_j.first; j < bounds_j.second; ++j) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; ++i) {
                        int i_pred = i - 1;
                        int j_pred = j - 1;
                        int k_pred = k - 1;

                        applyPeriodicBoundary(i_pred, j_pred, k_pred);

                        i_pred = i_pred + j * Ni + k * Ni * Nj;
                        j_pred = i + j_pred * Ni + k * Ni * Nj;
                        k_pred = i + j * Ni + k_pred * Ni * Nj;

                        int index = i + j * Ni + k * Ni * Nj;

                        Ex[index] += coef_J * Jx[index] + 
                            coef_E_dy * (Bz[index] - Bz[j_pred]) - 
                            coef_E_dz * (By[index] - By[k_pred]);
                        Ey[index] += coef_J * Jy[index] + 
                            coef_E_dz * (Bx[index] - Bx[k_pred]) - 
                            coef_E_dx * (Bz[index] - Bz[i_pred]);
                        Ez[index] += coef_J * Jz[index] + 
                            coef_E_dx * (By[index] - By[i_pred]) - 
                            coef_E_dy * (Bx[index] - Bx[j_pred]);
                    }
                }
            }
        }

        inline void update_B (Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
            int Ni = parameters.Ni;
            int Nj = parameters.Nj;
            int Nk = parameters.Nk;
            
            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; ++k) {
                for (int j = bounds_j.first; j < bounds_j.second; ++j) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; ++i) {
                        int i_next = i + 1;
                        int j_next = j + 1;
                        int k_next = k + 1;

                        applyPeriodicBoundary(i_next, j_next, k_next);

                        i_next = i_next + j * Ni + k * Ni * Nj;
                        j_next = i + j_next * Ni + k * Ni * Nj;
                        k_next = i + j * Ni + k_next * Ni * Nj;

                        int index = i + j * Ni + k * Ni * Nj;

                        Bx[index] += coef_B_dz * (Ey[k_next] - Ey[index]) - 
                            coef_B_dy * (Ez[j_next] - Ez[index]);
                        By[index] += coef_B_dx * (Ez[i_next] - Ez[index]) - 
                            coef_B_dz * (Ex[k_next] - Ex[index]);
                        Bz[index] += coef_B_dy * (Ex[j_next] - Ex[index]) - 
                            coef_B_dx * (Ey[i_next] - Ey[index]);
                    }
                }
            }
        }

    public:
        FDTD(Parameters _parameters) : parameters(_parameters) {
            if (parameters.Ni <= 0 || 
                parameters.Nj <= 0 || 
                parameters.Nk <= 0 || 
                parameters.dt <= 0) {
                throw std::invalid_argument("ERROR: invalid parameters");
            }

            size = parameters.Nk * parameters.Nj * parameters.Ni;

            Jx = Field(size, 0.0);
            Jy = Field(size, 0.0);
            Jz = Field(size, 0.0);
            Ex = Field(size, 0.0);
            Ey = Field(size, 0.0);
            Ez = Field(size, 0.0);
            Bx = Field(size, 0.0);
            By = Field(size, 0.0);
            Bz = Field(size, 0.0);

            size_i_main = { 0, parameters.Ni };
            size_j_main = { 0, parameters.Nj };
            size_k_main = { 0, parameters.Nk };

            coef_J = -4.0 * FDTD_const::PI * parameters.dt;

            coef_E_dx = FDTD_const::C * parameters.dt / parameters.dx;
            coef_E_dy = FDTD_const::C * parameters.dt / parameters.dy;
            coef_E_dz = FDTD_const::C * parameters.dt / parameters.dz;

            coef_B_dx = FDTD_const::C * parameters.dt / (2.0 * parameters.dx);
            coef_B_dy = FDTD_const::C * parameters.dt / (2.0 * parameters.dy);
            coef_B_dz = FDTD_const::C * parameters.dt / (2.0 * parameters.dz);
        }

        Field& get_field(Component this_field) {
            switch (this_field) {
                case Component::EX: return Ex;
                case Component::EY: return Ey;
                case Component::EZ: return Ez;
                case Component::BX: return Bx;
                case Component::BY: return By;
                case Component::BZ: return Bz;
                case Component::JX: return Jx;
                case Component::JY: return Jy;
                case Component::JZ: return Jz;
                default: throw std::logic_error("ERROR: Invalid component");
            }
        }

        inline void zeroes_currents() {
            Jx = Field(size, 0.0);
            Jy = Field(size, 0.0);
            Jz = Field(size, 0.0);
        }
        
        virtual inline void update_fields() {
            update_B(size_i_main, size_j_main, size_k_main);
        
            update_E(size_i_main, size_j_main, size_k_main);
        
            update_B(size_i_main, size_j_main, size_k_main);
        }
    };

}

