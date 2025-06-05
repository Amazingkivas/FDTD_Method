#include "FDTD_kokkos.h"

using namespace FDTD_kokkos;

FDTD::FDTD(Parameters _parameters, FP _dt) :
    parameters(_parameters), dt(_dt) {
    if (parameters.Ni <= 0 ||
        parameters.Nj <= 0 ||
        parameters.Nk <= 0 ||
        dt <= 0) {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    const int size = parameters.Ni * parameters.Nj * parameters.Nk;

    Jx = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Jx"), size);
    Jy = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Jy"), size);
    Jz = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Jz"), size);
    Ex = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Ex"), size);
    Ey = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Ey"), size);
    Ez = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Ez"), size);
    Bx = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Bx"), size);
    By = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "By"), size);
    Bz = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Bz"), size);

    Kokkos::parallel_for("FirstTouch", Kokkos::RangePolicy<>(0, size),
        KOKKOS_LAMBDA(int i) {
        Ex(i) = 0.0;
        Ey(i) = 0.0;
        Ez(i) = 0.0;
        Bx(i) = 0.0;
        By(i) = 0.0;
        Bz(i) = 0.0;
        Jx(i) = 0.0;
        Jy(i) = 0.0;
        Jz(i) = 0.0;
    });

    const FP cdt = FDTD_const::C * dt;

    coef_Ex = cdt / parameters.dx;
    coef_Ey = cdt / parameters.dy;
    coef_Ez = cdt / parameters.dz;

    coef_Bx = cdt / (2.0 * parameters.dx);
    coef_By = cdt / (2.0 * parameters.dy);
    coef_Bz = cdt / (2.0 * parameters.dz);
    
    current_coef = -4.0 * FDTD_const::PI * dt;

    Ni = parameters.Ni;
    Nj = parameters.Nj;
    Nk = parameters.Nk;

    begin_main_i = 0;
    begin_main_j = 0; 
    begin_main_k = 0;
    end_main_i = parameters.Ni;
    end_main_j = parameters.Nj;
    end_main_k = parameters.Nk;
}

void FDTD::zeroed_currents() {
    Kokkos::parallel_for("ZeroJx", Jx.extent(0), KOKKOS_LAMBDA(const int i) {
        Jx(i) = 0.0;
    });
    Kokkos::parallel_for("ZeroJy", Jx.extent(0), KOKKOS_LAMBDA(const int i) {
        Jy(i) = 0.0;
    });
    Kokkos::parallel_for("ZeroJz", Jx.extent(0), KOKKOS_LAMBDA(const int i) {
        Jz(i) = 0.0;
    });
}

Field& FDTD::get_field(Component this_field) {
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

void FDTD::update_fields() {
    const int size_i_main[2] = { 0, parameters.Ni };
    const int size_j_main[2] = { 0, parameters.Nj };
    const int size_k_main[2] = { 0, parameters.Nk };

    ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
    size_i_main, size_j_main, size_k_main, Ni, Nj, Nk,
    coef_Bx, coef_By, coef_Bz);

    ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
    Jx, Jy, Jz, current_coef,
    size_i_main, size_j_main, size_k_main, Ni, Nj, Nk,
    coef_Ex, coef_Ey, coef_Ez);

    ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
    size_i_main, size_j_main, size_k_main, Ni, Nj, Nk,
    coef_Bx, coef_By, coef_Bz);
}
