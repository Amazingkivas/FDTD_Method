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

    Jx = Field("Jx", size);
    Jy = Field("Jy", size);
    Jz = Field("Jz", size);
    Ex = Field("Ex", size);
    Ey = Field("Ey", size);
    Ez = Field("Ez", size);
    Bx = Field("Bx", size);
    By = Field("By", size);
    Bz = Field("Bz", size);

    size_i_main[0] = 0;
    size_i_main[1] = parameters.Ni;
        
    size_j_main[0] = 0;
    size_j_main[1] = parameters.Nj;
        
    size_k_main[0] = 0;
    size_k_main[1] = parameters.Nk;

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
    ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
    size_i_main, size_j_main, size_k_main, Ni, Nj, Nk,
    parameters.dx, parameters.dy, parameters.dz, dt);

    ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
    Jx, Jy, Jz, current_coef,
    size_i_main, size_j_main, size_k_main, Ni, Nj, Nk,
    parameters.dx, parameters.dy, parameters.dz, dt);

    ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
    size_i_main, size_j_main, size_k_main, Ni, Nj, Nk,
    parameters.dx, parameters.dy, parameters.dz, dt);
}
