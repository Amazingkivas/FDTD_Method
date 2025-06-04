#include "FDTD_PML_kokkos.h"

void FDTD_kokkos::FDTD_PML::update_B_PML(Boundaries bounds_i,
    Boundaries bounds_j, Boundaries bounds_k) {
    ComputeB_PML_FieldFunctor::apply(Ex, Ey, Ez, Bxy,
        Byx, Bzy, Byz, Bzx, Bxz,
        Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
        this->dt, this->parameters.dx, this->parameters.dy, this->parameters.dz,
        bounds_i, bounds_j, bounds_k,
        this->parameters.Ni, this->parameters.Nj, this->parameters.Nk);
}
void FDTD_kokkos::FDTD_PML::update_E_PML(Boundaries bounds_i,
    Boundaries bounds_j, Boundaries bounds_k) {
    ComputeE_PML_FieldFunctor::apply(Ex, Ey, Ez, Exy,
        Eyx, Ezy, Eyz, Ezx, Exz,
        Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
        this->dt, this->parameters.dx, this->parameters.dy, this->parameters.dz,
        bounds_i, bounds_j, bounds_k,
        this->parameters.Ni, this->parameters.Nj, this->parameters.Nk);
}

FDTD_kokkos::FDTD_PML::FDTD_PML(Parameters _parameters, FP _dt, FP pml_percent) :
    FDTD(_parameters, _dt) {
    const int size = _parameters.Ni * _parameters.Nj * _parameters.Nk;

    Exy = Field("Exy", size);
    Exz = Field("Exz", size);
    Eyx = Field("Eyx", size);
    Eyz = Field("Eyz", size);
    Ezx = Field("Ezx", size);
    Ezy = Field("Ezy", size);

    Bxy = Field("Bxy", size);
    Bxz = Field("Bxz", size);
    Byx = Field("Byx", size);
    Byz = Field("Byz", size);
    Bzx = Field("Bzx", size);
    Bzy = Field("Bzy", size);

    EsigmaX = Field("EsigmaX", size);
    EsigmaY = Field("EsigmaY", size);
    EsigmaZ = Field("EsigmaZ", size);
    BsigmaX = Field("BsigmaX", size);
    BsigmaY = Field("BsigmaY", size);
    BsigmaZ = Field("BsigmaZ", size);

    pml_size_i = static_cast<int>(static_cast<FP>(_parameters.Ni) * pml_percent);
    pml_size_j = static_cast<int>(static_cast<FP>(_parameters.Nj) * pml_percent);
    pml_size_k = static_cast<int>(static_cast<FP>(_parameters.Nk) * pml_percent);

    // Defining areas of computation
    // ======================================================================
    size_i_main = { pml_size_i, _parameters.Ni - pml_size_i };
    size_j_main = { pml_size_j, _parameters.Nj - pml_size_j };
    size_k_main = { pml_size_k, _parameters.Nk - pml_size_k };

    this->begin_main_i = size_i_main.first;
    this->begin_main_j = size_j_main.first; 
    this->begin_main_k = size_k_main.first;
    this->end_main_i = size_i_main.second;
    this->end_main_j = size_j_main.second;
    this->end_main_k = size_k_main.second;

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
    FP SGm_x = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<FP>(pml_size_i) * _parameters.dx);
    FP SGm_y = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<FP>(pml_size_j) * _parameters.dy);
    FP SGm_z = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<FP>(pml_size_k) * _parameters.dz);
    // ======================================================================

    // Calculation of permittivity and permeability in the cells
    // ======================================================================
    ComputeSigmaFunctor::apply(EsigmaZ, BsigmaZ, SGm_z, calc_distant_k_low, pml_size_k, dt,
        size_i_solid, size_j_solid, size_xy_lower_k_pml, Ni, Nj, Nk);
    ComputeSigmaFunctor::apply(EsigmaY, BsigmaY, SGm_y, calc_distant_j_low, pml_size_j, dt,
        size_i_solid, size_zx_lower_j_pml, size_k_solid, Ni, Nj, Nk);
    ComputeSigmaFunctor::apply(EsigmaX, BsigmaX, SGm_x, calc_distant_i_low, pml_size_i, dt,
        size_yz_lower_i_pml, size_j_solid, size_k_solid, Ni, Nj, Nk);

    ComputeSigmaFunctor::apply(EsigmaZ, BsigmaZ, SGm_z, calc_distant_k_up, pml_size_k, dt,
        size_i_solid, size_j_solid, size_xy_upper_k_pml, Ni, Nj, Nk);
    ComputeSigmaFunctor::apply(EsigmaY, BsigmaY, SGm_y, calc_distant_j_up, pml_size_j, dt,
        size_i_solid, size_zx_upper_j_pml, size_k_solid, Ni, Nj, Nk);
    ComputeSigmaFunctor::apply(EsigmaX, BsigmaX, SGm_x, calc_distant_i_up, pml_size_i, dt,
        size_yz_upper_i_pml, size_j_solid, size_k_solid, Ni, Nj, Nk);
    // ======================================================================
};

void FDTD_kokkos::FDTD_PML::update_fields() {
    int i_main[2] = { this->begin_main_i, this->end_main_i };
    int j_main[2] = { this->begin_main_j, this->end_main_j };
    int k_main[2] = { this->begin_main_k, this->end_main_k };

    ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
        i_main, j_main, k_main, Ni, Nj, Nk, coef_Bx, coef_By, coef_Bz);

    update_B_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
    update_B_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
    update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

    update_B_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
    update_B_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
    update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

    ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
        Jx, Jx, Jx, current_coef,
        i_main, j_main, k_main, Ni, Nj, Nk, coef_Ex, coef_Ey, coef_Ez);

    update_E_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
    update_E_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
    update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

    update_E_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
    update_E_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
    update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

    ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
        i_main, j_main, k_main, Ni, Nj, Nk, coef_Bx, coef_By, coef_Bz);
}
