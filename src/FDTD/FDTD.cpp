#include "FDTD.h"

FDTD_openmp::FDTD::FDTD(Parameters _parameters, FP _dt)
    : parameters(_parameters), dt(_dt) {
    if (parameters.Ni <= 0 || parameters.Nj <= 0 || parameters.Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    const int size = parameters.Nk * parameters.Nj * parameters.Ni;

    Jx = Field(size, 0.0);
    Jy = Field(size, 0.0);
    Jz = Field(size, 0.0);
    Ex = Field(size, 0.0);
    Ey = Field(size, 0.0);
    Ez = Field(size, 0.0);
    Bx = Field(size, 0.0);
    By = Field(size, 0.0);
    Bz = Field(size, 0.0);

    Ni = parameters.Ni;
    Nj = parameters.Nj;
    Nk = parameters.Nk;

    dx = parameters.dx;
    dy = parameters.dy;
    dz = parameters.dz;
    dt = _dt;

    const FP cdt = FDTD_const::C * dt;

    coef_E_dx = cdt / dx;
    coef_E_dy = cdt / dy;
    coef_E_dz = cdt / dz;

    coef_B_dx = cdt / (2.0 * dx);
    coef_B_dy = cdt / (2.0 * dy);
    coef_B_dz = cdt / (2.0 * dz);

    cur_coef = -4.0 * FDTD_const::PI * dt;

    begin_main_i = 0;
    begin_main_j = 0; 
    begin_main_k = 0;
    end_main_i = parameters.Ni;
    end_main_j = parameters.Nj;
    end_main_k = parameters.Nk;
}

void FDTD_openmp::FDTD::update_E() {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = begin_main_k; k < end_main_k; k++) {
        for (int j = begin_main_j; j < end_main_j; j++) {
            for (int i = begin_main_i; i < end_main_i; i++) {
                int i_pred = i - 1;
                int j_pred = j - 1;
                int k_pred = k - 1;

                applyPeriodicBoundary(i_pred, Ni);
                applyPeriodicBoundary(j_pred, Nj);
                applyPeriodicBoundary(k_pred, Nk);

                int index = i + j * Ni + k * Ni * Nj;
                i_pred = i_pred + j * Ni + k * Ni * Nj;
                j_pred = i + j_pred * Ni + k * Ni * Nj;
                k_pred = i + j * Ni + k_pred * Ni * Nj;

                Ex[index] += cur_coef * Jx[index] + 
                FDTD_const::C * dt * ((Bz[index] - Bz[j_pred]) / dy -
                (By[index] - By[k_pred]) / dz);
                Ey[index] += cur_coef * Jx[index] + 
                FDTD_const::C * dt * ((Bx[index] - Bx[k_pred]) / dz -
                (Bz[index] - Bz[i_pred]) / dx);
                Ez[index] += cur_coef * Jx[index] + 
                FDTD_const::C * dt * ((By[index] - By[i_pred]) / dx -
                (Bx[index] - Bx[j_pred]) / dy);
            }
        }
    }
}

void FDTD_openmp::FDTD::update_B() {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = begin_main_k; k < end_main_k; k++) {
        for (int j = begin_main_j; j < end_main_j; j++) {
            for (int i = begin_main_i; i < end_main_i; i++) {
                int i_next = i + 1;
                int j_next = j + 1;
                int k_next = k + 1;

                applyPeriodicBoundary(i_next, Ni);
                applyPeriodicBoundary(j_next, Nj);
                applyPeriodicBoundary(k_next, Nk);

                int index = i + j * Ni + k * Ni * Nj;
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
        }
    }
}

void FDTD_openmp::FDTD::zeroed_currents() {
    std::fill(Jx.begin(), Jx.end(), 0.0);
    std::fill(Jy.begin(), Jy.end(), 0.0);
    std::fill(Jz.begin(), Jz.end(), 0.0);
}

FDTD_openmp::Field& FDTD_openmp::FDTD::get_field(Component this_field) {
    switch (this_field) {
        case Component::JX: return Jx;
        case Component::JY: return Jy;
        case Component::JZ: return Jz;
        case Component::EX: return Ex;
        case Component::EY: return Ey;
        case Component::EZ: return Ez;
        case Component::BX: return Bx;
        case Component::BY: return By;
        case Component::BZ: return Bz;
        default: throw std::logic_error("ERROR: Invalid field component");
    }
}

void FDTD_openmp::FDTD::update_fields() {
    update_B();
    update_E();
    update_B();
}
