#pragma once

#include "FDTD.h"

namespace FDTD_openmp {

    class FDTD_PML : public FDTD {
    private:
        Field Exy, Exz, Eyx, Eyz, Ezx, Ezy;
        Field Bxy, Bxz, Byx, Byz, Bzx, Bzy;
        Field EsigmaX, EsigmaY, EsigmaZ;
        Field BsigmaX, BsigmaY, BsigmaZ;

        Boundaries size_i_main, size_j_main, size_k_main;
        Boundaries size_i_solid, size_j_solid, size_k_solid;
        Boundaries size_i_part_from_start, size_i_part_from_end,
            size_k_part_from_start, size_k_part_from_end,
            size_xy_lower_k_pml, size_xy_upper_k_pml,
            size_yz_lower_i_pml, size_yz_upper_i_pml,
            size_zx_lower_j_pml, size_zx_upper_j_pml;

        int pml_size_i, pml_size_j, pml_size_k;

        inline void set_sigma_x(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
            double SGm, Function dist) {
            int Ni = this->parameters.Ni;
            int Nj = this->parameters.Nj;
            int Nk = this->parameters.Nk;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; k++) {
                for (int j = bounds_j.first; j < bounds_j.second; j++) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; i++) {
                        int index = i + j * Ni + k * Ni * Nj;

                        EsigmaX[index] = SGm *
                            std::pow((static_cast<double>(dist[index])) /
                            static_cast<double>(pml_size_i), FDTDconst::N);
                        BsigmaX[index] = SGm *
                            std::pow((static_cast<double>(dist[index])) /
                            static_cast<double>(pml_size_i), FDTDconst::N);
                    }
                }
            }
        }
        inline void set_sigma_y(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
            double SGm, Function dist) {
            int Ni = this->parameters.Ni;
            int Nj = this->parameters.Nj;
            int Nk = this->parameters.Nk;
            
            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; k++) {
                for (int j = bounds_j.first; j < bounds_j.second; j++) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; i++) {
                        int index = i + j * Ni + k * Ni * Nj;

                        EsigmaY[index] = SGm *
                            std::pow((static_cast<double>(dist[index])) /
                            static_cast<double>(pml_size_j), FDTDconst::N);
                        BsigmaY[index] = SGm *
                            std::pow((static_cast<double>(dist[index])) /
                            static_cast<double>(pml_size_j), FDTDconst::N);
                    }
                }
            }
        }
        inline void set_sigma_z(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
            double SGm, Function dist) {
            int Ni = this->parameters.Ni;
            int Nj = this->parameters.Nj;
            int Nk = this->parameters.Nk;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; k++) {
                for (int j = bounds_j.first; j < bounds_j.second; j++) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; i++) {
                        int index = i + j * Ni + k * Ni * Nj;

                        EsigmaZ[index] = SGm *
                            std::pow((static_cast<double>(dist[index])) /
                            static_cast<double>(pml_size_k), FDTDconst::N);
                        BsigmaZ[index] = SGm *
                            std::pow((static_cast<double>(dist[index])) /
                            static_cast<double>(pml_size_k), FDTDconst::N);
                    }
                }
            }
        }

        inline double PMLcoef(double sigma) const {
            return std::exp(-sigma * this->parameters.dt * FDTD_const::C);
        }

        inline void update_E_PML(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
            double dx = this->parameters.dx;
            double dy = this->parameters.dy;
            double dz = this->parameters.dz;

            int Ni = this->parameters.Ni;
            int Nj = this->parameters.Nj;
            int Nk = this->parameters.Nk;

            double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; k++)  {
                for (int j = bounds_j.first; j < bounds_j.second; j++) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; i++) {
                        int i_pred = i - 1;
                        int j_pred = j - 1;
                        int k_pred = k - 1;

                        this->applyPeriodicBoundary(i_pred, j_pred, k_pred);

                        int index = i + j * Ni + k * Ni * Nj;

                        i_pred = i_pred + j * Ni + k * Ni * Nj;
                        j_pred = i + j_pred * Ni + k * Ni * Nj;
                        k_pred = i + j * Ni + k_pred * Ni * Nj;

                        if (EsigmaX[index] != 0.0)
                            PMLcoef2_x = (1.0 - PMLcoef(EsigmaX[index])) / (EsigmaX[index] * dx);
                        else
                            PMLcoef2_x = this->coef_E_dx;

                        if (EsigmaY[index] != 0.0)
                            PMLcoef2_y = (1.0 - PMLcoef(EsigmaY[index])) / (EsigmaY[index] * dy);
                        else
                            PMLcoef2_y = this->coef_E_dy;

                        if (EsigmaZ[index] != 0.0)
                            PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ[index])) / (EsigmaZ[index] * dz);
                        else
                            PMLcoef2_z = this->coef_E_dz;

                        Eyx[index] = Eyx[index] * PMLcoef(EsigmaX[index]) -
                            PMLcoef2_x * (this->Bz[index] - this->Bz[i_pred]);
                        Ezx[index] = Ezx[index] * PMLcoef(EsigmaX[index]) +
                            PMLcoef2_x * (this->By[index] - this->By[i_pred]);

                        Exy[index] = Exy[index] * PMLcoef(EsigmaY[index]) +
                            PMLcoef2_y * (this->Bz[index] - this->Bz[j_pred]);
                        Ezy[index] = Ezy[index] * PMLcoef(EsigmaY[index]) -
                            PMLcoef2_y * (this->Bx[index] - this->Bx[j_pred]);

                        Exz[index] = Exz[index] * PMLcoef(EsigmaZ[index]) -
                            PMLcoef2_z * (this->By[index] - this->By[k_pred]);
                        Eyz[index] = Eyz[index] * PMLcoef(EsigmaZ[index]) +
                            PMLcoef2_z * (this->Bx[index] - this->Bx[k_pred]);

                        this->Ex[index] = Exz[index] + Exy[index];
                        this->Ey[index] = Eyx[index] + Eyz[index];
                        this->Ez[index] = Ezy[index] + Ezx[index];
                    }
                }
            }
        }

        inline void update_B_PML(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k) {
            double dx = this->parameters.dx;
            double dy = this->parameters.dy;
            double dz = this->parameters.dz;

            int Ni = this->parameters.Ni;
            int Nj = this->parameters.Nj;
            int Nk = this->parameters.Nk;

            double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = bounds_k.first; k < bounds_k.second; k++) {
                for (int j = bounds_j.first; j < bounds_j.second; j++) {
                    #pragma omp simd
                    for (int i = bounds_i.first; i < bounds_i.second; i++) {
                        int i_next = i + 1;
                        int j_next = j + 1;
                        int k_next = k + 1;

                        this->applyPeriodicBoundary(i_next, j_next, k_next);

                        int index = i + j * Ni + k * Ni * Nj;

                        i_next = i_next + j * Ni + k * Ni * Nj;
                        j_next = i + j_next * Ni + k * Ni * Nj;
                        k_next = i + j * Ni + k_next * Ni * Nj;

                        if (BsigmaX[index] != 0.0)
                            PMLcoef2_x = (1.0 - PMLcoef(BsigmaX[index])) / (BsigmaX[index] * dx);
                        else
                            PMLcoef2_x = this->coef_E_dx;

                        if (BsigmaY[index] != 0.0)
                            PMLcoef2_y = (1.0 - PMLcoef(BsigmaY[index])) / (BsigmaY[index] * dy);
                        else
                            PMLcoef2_y = this->coef_E_dy;

                        if (BsigmaZ[index] != 0.0)
                            PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ[index])) / (BsigmaZ[index] * dz);
                        else
                            PMLcoef2_z = this->coef_E_dz;

                        Byx[index] = Byx[index] * PMLcoef(BsigmaX[index]) +
                            PMLcoef2_x * (this->Ez[i_next] - this->Ez[index]);
                        Bzx[index] = Bzx[index] * PMLcoef(BsigmaX[index]) -
                            PMLcoef2_x * (this->Ey[i_next] - this->Ey[index]);

                        Bxy[index] = Bxy[index] * PMLcoef(BsigmaY[index]) -
                            PMLcoef2_y * (this->Ez[j_next] - this->Ez[index]);
                        Bzy[index] = Bzy[index] * PMLcoef(BsigmaY[index]) +
                            PMLcoef2_y * (this->Ex[j_next] - this->Ex[index]);

                        Bxz[index] = Bxz[index] * PMLcoef(BsigmaZ[index]) +
                            PMLcoef2_z * (this->Ey[k_next] - this->Ey[index]);
                        Byz[index] = Byz[index] * PMLcoef(BsigmaZ[index]) -
                            PMLcoef2_z * (this->Ex[k_next] - this->Ex[index]);

                        this->Bx[index] = Bxy[index] + Bxz[index];
                        this->By[index] = Byz[index] + Byx[index];
                        this->Bz[index] = Bzx[index] + Bzy[index];
                    }
                }
            }
        }

    public:
        FDTD_PML(Parameters _parameters, int pml_percent) : FDTD(_parameters) {
            int size = _parameters.Ni * _parameters.Nj * _parameters.Nk

            Exy = Exz = Eyx = Eyz = Ezx = Ezy = Field(size, 0.0);
            Bxy = Bxz = Byx = Byz = Bzx = Bzy = Field(size, 0.0);
            EsigmaX = EsigmaY = EsigmaZ = BsigmaX = BsigmaY = BsigmaZ = Field(size, 0.0);

            pml_size_i = static_cast<int>(static_cast<double>(_parameters.Ni) * pml_percent);
            pml_size_j = static_cast<int>(static_cast<double>(_parameters.Nj) * pml_percent);
            pml_size_k = static_cast<int>(static_cast<double>(_parameters.Nk) * pml_percent);

            // Defining areas of computation
            // ======================================================================
            size_i_main = { pml_size_i, _parameters.Ni - pml_size_i };
            size_j_main = { pml_size_j, _parameters.Nj - pml_size_j };
            ize_k_main = { pml_size_k, _parameters.Nk - pml_size_k };

            size_i_solid = { 0, _parameters.Ni };
            size_j_solid = { 0, _parameters.Nj };
            size_k_solid = { 0, _parameters.Nk };

            size_i_part_from_start = { 0, _parameters.Ni - pml_size_i };
            size_i_part_from_end = { pml_size_i, _parameters.Ni };

            size_k_part_from_start = { 0, _parameters.Nk - pml_size_k };
            size_k_part_from_end = { pml_size_k, _parameters.Nk };

            size_xy_lower_k_pml = { 0, pml_size_k };
            size_xy_upper_k_pml = { _parameters.Nk - pml_size_k, _parameters.Nk };

            size_yz_lower_i_pml = { 0, pml_size_i };
            size_yz_upper_i_pml = { _parameters.Ni - pml_size_i, _parameters.Ni };

            size_zx_lower_j_pml = { 0, pml_size_j };
            size_zx_upper_j_pml = { _parameters.Nj - pml_size_j, _parameters.Nj };
            // ======================================================================

            // Definition of functions for calculating the distance to the interface
            // ======================================================================
            Function calc_distant_i_up = [=](int i, int j, int k) {
                return i + 1 + pml_size_i - _parameters.Ni;
            };
            Function calc_distant_j_up = [=](int i, int j, int k) {
                return j + 1 + pml_size_j - _parameters.Nj;
            };
            Function calc_distant_k_up = [=](int i, int j, int k) {
                return k + 1 + pml_size_k - _parameters.Nk;
            };

            Function calc_distant_i_low = [=](int i, int j, int k) {
                return pml_size_i - i;
            };
            Function calc_distant_j_low = [=](int i, int j, int k) {
                return pml_size_j - j;
            };
            Function calc_distant_k_low = [=](int i, int j, int k) {
                return pml_size_k - k;
            };
            // ======================================================================

            // Calculation of maximum permittivity and permeability
            // ======================================================================
            double SGm_x = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
                / (static_cast<double>(pml_size_i) * _parameters.dx);
            double SGm_y = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
                / (static_cast<double>(pml_size_j) * _parameters.dy);
            double SGm_z = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
                / (static_cast<double>(pml_size_k) * _parameters.dz);
            // ======================================================================

            // Calculation of permittivity and permeability in the cells
            // ======================================================================
            set_sigma_z(size_i_solid, size_j_solid, size_xy_lower_k_pml,
                SGm_z, calc_distant_k_low);
            set_sigma_y(size_i_solid, size_zx_lower_j_pml, size_k_solid,
                SGm_y, calc_distant_j_low);
            set_sigma_x(size_yz_lower_i_pml, size_j_solid, size_k_solid,
                SGm_x, calc_distant_i_low);

            set_sigma_z(size_i_solid, size_j_solid, size_xy_upper_k_pml,
                SGm_z, calc_distant_k_up);
            set_sigma_y(size_i_solid, size_zx_upper_j_pml, size_k_solid,
                SGm_y, calc_distant_j_up);
            set_sigma_x(size_yz_upper_i_pml, size_j_solid, size_k_solid,
                SGm_x, calc_distant_i_up);
            // ======================================================================
        }

        inline void update_fields() override {
            this->update_B(size_i_main, size_j_main, size_k_main);

            update_B_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
            update_B_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
            update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

            update_B_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
            update_B_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
            update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

            this->update_E(size_i_main, size_j_main, size_k_main);

            update_E_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
            update_E_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
            update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

            update_E_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
            update_E_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
            update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

            this->update_B(size_i_main, size_j_main, size_k_main);
        }
    };

}
