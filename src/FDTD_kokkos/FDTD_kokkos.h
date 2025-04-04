#pragma once

#include "kokkos_shared.h"
#include "Structures.h"

namespace FDTD_kokkos {

    using namespace FDTD_struct;

    class FDTD {
    private:
        Boundaries size_i_main, size_j_main, size_k_main;

    protected:
        Parameters parameters;
        Field Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz;

        double coef_J;
        double coef_E_dx, coef_E_dy, coef_E_dz;
        double coef_B_dx, coef_B_dy, coef_B_dz;

        KOKKOS_INLINE_FUNCTION
        void applyPeriodicBoundary(int& i, int& j, int& k) {
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

        void update_E (Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
            int Ni = parameters.Ni;
            int Nj = parameters.Nj;
            int Nk = parameters.Nk;

            int begin = bounds_i.first +
                bounds_j.first * Ni +
                bounds_k.first * Ni * Nj;
            int end = bounds_i.second - 1 +
                (bounds_j.second - 1) * Ni +
                (bounds_k.second - 1) * Ni * Nj;

            using simd_type = Kokkos::Experimental::native_simd<double>;
            constexpr int simd_width = simd_type::size();

            Kokkos::parallel_for("UpdateEField", Kokkos::RangePolicy<>(begin, end / simd_width),
                KOKKOS_LAMBDA(const int i) {
                simd_type Ex_simd, Ey_simd, Ez_simd;
                simd_type Bx_simd, By_simd, Bz_simd;
                simd_type Bz_pred_simd, By_pred_simd, Bx_pred_simd;
                simd_type Bz_i_pred_simd, By_i_pred_simd, Bx_j_pred_simd;

                int base_index = i * simd_width;

                for (int lane = 0; lane < simd_width; ++lane) {
                    int index = base_index + lane;

                    int k = index / (Ni * Nj);
                    int j = index / Ni - k * Nj;
                    int i_idx = index - j * Ni - k * Ni * Nj;

                    int i_pred = i_idx - 1;
                    int j_pred = j - 1;
                    int k_pred = k - 1;

                    applyPeriodicBoundary(i_pred, j_pred, k_pred);

                    i_pred = i_pred + j * Ni + k * Ni * Nj;
                    j_pred = i_idx + j_pred * Ni + k * Ni * Nj;
                    k_pred = i_idx + j * Ni + k_pred * Ni * Nj;

                    Ex_simd[lane] = Ex[index];
                    Ey_simd[lane] = Ey[index];
                    Ez_simd[lane] = Ez[index];

                    Bx_simd[lane] = Bx[index];
                    By_simd[lane] = By[index];
                    Bz_simd[lane] = Bz[index];

                    Bz_pred_simd[lane] = Bz[j_pred];
                    By_pred_simd[lane] = By[k_pred];
                    Bx_pred_simd[lane] = Bx[k_pred];
                    Bz_i_pred_simd[lane] = Bz[i_pred];
                    By_i_pred_simd[lane] = By[i_pred];
                    Bx_j_pred_simd[lane] = Bx[j_pred];
                }

                Ex_simd += coef_J * Jx(base_index) + 
                    coef_E_dy * (Bz_simd - Bz_pred_simd) -
                    coef_E_dz * (By_simd - By_pred_simd);
                Ey_simd += coef_J * Jy(base_index) + 
                    coef_E_dz * (Bx_simd - Bx_pred_simd) -
                    coef_E_dx * (Bz_simd - Bz_i_pred_simd);
                Ez_simd += coef_J * Jz(base_index) + 
                    coef_E_dx * (By_simd - By_i_pred_simd) -
                    coef_E_dy * (Bx_simd - Bx_j_pred_simd);

                for (int lane = 0; lane < simd_width; ++lane) {
                    int index = base_index + lane;
                    Ex[index] = Ex_simd[lane];
                    Ey[index] = Ey_simd[lane];
                    Ez[index] = Ez_simd[lane];
                }
            });
        }

        void update_B (Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
            int Ni = parameters.Ni;
            int Nj = parameters.Nj;
            int Nk = parameters.Nk;
            
            int begin = bounds_i.first +
                bounds_j.first * Ni +
                bounds_k.first * Ni * Nj;
            int end = bounds_i.second - 1 +
                (bounds_j.second - 1) * Ni +
                (bounds_k.second - 1) * Ni * Nj;

            using simd_type = Kokkos::Experimental::native_simd<double>;
            constexpr int simd_width = simd_type::size();

            Kokkos::parallel_for("UpdateBField", Kokkos::RangePolicy<>(begin, end / simd_width),
                KOKKOS_LAMBDA(const int i) {
                simd_type Bx_simd, By_simd, Bz_simd;
                simd_type Ex_simd, Ey_simd, Ez_simd;
                simd_type Ey_next_simd, Ez_next_simd, Ex_next_simd;
                simd_type Ey_i_next_simd, Ez_j_next_simd, Ex_k_next_simd;

                int base_index = i * simd_width;

                for (int lane = 0; lane < simd_width; ++lane) {
                    int index = base_index + lane;

                    int k = index / (Ni * Nj);
                    int j = index / Ni - k * Nj;
                    int i_idx = index - j * Ni - k * Ni * Nj;

                    int i_next = i_idx + 1;
                    int j_next = j + 1;
                    int k_next = k + 1;

                    applyPeriodicBoundary(i_next, j_next, k_next);

                    i_next = i_next + j * Ni + k * Ni * Nj;
                    j_next = i_idx + j_next * Ni + k * Ni * Nj;
                    k_next = i_idx + j * Ni + k_next * Ni * Nj;

                    Bx_simd[lane] = Bx[index];
                    By_simd[lane] = By[index];
                    Bz_simd[lane] = Bz[index];

                    Ex_simd[lane] = Ex[index];
                    Ey_simd[lane] = Ey[index];
                    Ez_simd[lane] = Ez[index];

                    Ey_next_simd[lane] = Ey[k_next];
                    Ez_next_simd[lane] = Ez[i_next];
                    Ex_next_simd[lane] = Ex[j_next];

                    Ey_i_next_simd[lane] = Ey[i_next];
                    Ez_j_next_simd[lane] = Ez[j_next];
                    Ex_k_next_simd[lane] = Ex[k_next];
                }

                Bx_simd += coef_B_dz * (Ey_next_simd - Ey_simd) -
                    coef_B_dy *(Ez_j_next_simd - Ez_simd);
                By_simd += coef_B_dx * (Ez_next_simd - Ez_simd) -
                    coef_B_dz *(Ex_k_next_simd - Ex_simd);
                Bz_simd += coef_B_dy * (Ex_next_simd - Ex_simd) -
                    coef_B_dx *(Ey_i_next_simd - Ey_simd);

                for (int lane = 0; lane < simd_width; ++lane) {
                    int index = base_index + lane;
                    Bx[index] = Bx_simd[lane];
                    By[index] = By_simd[lane];
                    Bz[index] = Bz_simd[lane];
                }
            });
        }

    public:
        FDTD(Parameters _parameters) : parameters(_parameters) {
            if (parameters.Ni <= 0 ||
                parameters.Nj <= 0 ||
                parameters.Nk <= 0 ||
                parameters.dt <= 0) {
                throw std::invalid_argument("ERROR: invalid parameters");
            }

            int size = parameters.Ni * parameters.Nj * parameters.Nk;

            Jx = Field("Jx", size);
            Jy = Field("Jy", size);
            Jz = Field("Jz", size);
            Ex = Field("Ex", size);
            Ey = Field("Ey", size);
            Ez = Field("Ez", size);
            Bx = Field("Bx", size);
            By = Field("By", size);
            Bz = Field("Bz", size);

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
        };

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
            default: throw std::logic_error("ERROR: Invalid field component");
            }
        }

        void zeroes_currents() {
            Kokkos::deep_copy(Jx, 0.0);
            Kokkos::deep_copy(Jy, 0.0);
            Kokkos::deep_copy(Jz, 0.0);
        }

        virtual void update_fields() {
            update_B(size_i_main, size_j_main, size_k_main);
        
            update_E(size_i_main, size_j_main, size_k_main);
        
            update_B(size_i_main, size_j_main, size_k_main);
        }
    };

}
