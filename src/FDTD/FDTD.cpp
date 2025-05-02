#include "FDTD.h"

FDTD::FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar,
    std::function<double(double, double, double, double)> init_function)
    : parameters(_parameters), cParams(_Cpar), dt(_dt), pml_percent(_pml_percent), time(time_), init_func(init_function) {
    if (parameters.Ni <= 0 || parameters.Nj <= 0 || parameters.Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    int size = (parameters.Nk + 2) * (parameters.Nj + 2) * (parameters.Ni + 2);

    Jx = Field(size);
    Ex = Field(size);
    Ey = Field(size);
    Ez = Field(size);
    Bx = Field(size);
    By = Field(size);
    Bz = Field(size);

    pml_size_i = static_cast<int>(static_cast<double>(parameters.Ni) * pml_percent);
    pml_size_j = static_cast<int>(static_cast<double>(parameters.Nj) * pml_percent);
    pml_size_k = static_cast<int>(static_cast<double>(parameters.Nk) * pml_percent);
}

inline void FDTD::applyPeriodicBoundaryE() {
    int Ni = parameters.Ni + 2;
    int Nj = parameters.Nj + 2;
    int Nk = parameters.Nk + 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Ni - 2; ++i) {
        for (int j = 1; j <= Nj - 2; ++j) {
             int index_right = i + j * Ni + (parameters.Nk + 1) * Ni * Nj;
             int index_left = i + j * Ni + Ni * Nj;
             Ex[index_right] = Ex[index_left];
             Ey[index_right] = Ey[index_left];
             Ez[index_right] = Ez[index_left];
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Ni - 2; ++i) {
        for (int k = 1; k <= Nk - 2; ++k) {
             int index_right = i + (parameters.Nj + 1) * Ni + k * Ni * Nj;
             int index_left = i + Ni + k * Ni * Nj;
             Ex[index_right] = Ex[index_left];
             Ey[index_right] = Ey[index_left];
             Ez[index_right] = Ez[index_left];
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= Nk - 2; ++k) {
        for (int j = 1; j <= Nj - 2; ++j) {
             int index_right = parameters.Ni + 1 + j * Ni + k * Ni * Nj;
             int index_left = 1 + j * Ni + k * Ni * Nj;
             Ex[index_right] = Ex[index_left];
             Ey[index_right] = Ey[index_left];
             Ez[index_right] = Ez[index_left];
        }
    }

}

inline void FDTD::applyPeriodicBoundaryB() {
    int Ni = parameters.Ni + 2;
    int Nj = parameters.Nj + 2;
    int Nk = parameters.Nk + 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Ni - 2; ++i) {
        for (int j = 1; j <= Nj - 2; ++j) {
             int index_right = i + j * Ni + parameters.Nk * Ni * Nj;
             int index_left = i + j * Ni;
             Bx[index_left] = Bx[index_right];
             By[index_left] = By[index_right];
             Bz[index_left] = Bz[index_right];
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Ni - 2; ++i) {
        for (int k = 1; k <= Nk - 2; ++k) {
             int index_right = i + parameters.Nj * Ni + k * Ni * Nj;
             int index_left = i + k * Ni * Nj;
             Bx[index_left] = Bx[index_right];
             By[index_left] = By[index_right];
             Bz[index_left] = Bz[index_right];
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= Nk - 2; ++k) {
        for (int j = 1; j <= Nj - 2; ++j) {
             int index_right = parameters.Ni + j * Ni + k * Ni * Nj;
             int index_left = j * Ni + k * Ni * Nj;
             Bx[index_left] = Bx[index_right];
             By[index_left] = By[index_right];
             Bz[index_left] = Bz[index_right];
        }
    }
}

Field& FDTD::get_field(Component this_field) {
    switch (this_field) {
        case Component::EX: return Ex;
        case Component::EY: return Ey;
        case Component::EZ: return Ez;
        case Component::BX: return Bx;
        case Component::BY: return By;
        case Component::BZ: return Bz;
        default: throw std::logic_error("ERROR: Invalid field component");
    }
}

void FDTD::update_fields(bool write_result, Axis write_axis, std::string base_path) {
    if (time < 0) {
        throw std::invalid_argument("ERROR: Invalid update field argument");
    }

    int cur_time;

    if (init_func) {
    cur_time = cParams.iterations;

    double Tx = cParams.period_x;
    double Ty = cParams.period_y;
    double Tz = cParams.period_z;

    int start_i = static_cast<int>(floor((-Tx / 4.0 - parameters.ax) / parameters.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - parameters.ay) / parameters.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - parameters.az) / parameters.dz));

    int max_i = static_cast<int>(floor((Tx / 4.0 - parameters.ax) / parameters.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - parameters.ay) / parameters.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - parameters.az) / parameters.dz));

    int size_i_cur[2] = { start_i, max_i };
    int size_j_cur[2] = { start_j, max_j };
    int size_k_cur[2] = { start_k, max_k };

    //InitializeCurrentFunctor::apply(Jx, cParams, parameters, init_func, std::min(cur_time, time),
    //                                size_i_cur, size_j_cur, size_k_cur);
    //InitializeCurrentFunctor::apply(Jy, cParams, parameters, init_func, cur_time,
    //                                size_i_cur, size_j_cur, size_k_cur);
    //InitializeCurrentFunctor::apply(Jz, cParams, parameters, init_func, cur_time,
    //                                size_i_cur, size_j_cur, size_k_cur);
    } else {
    cur_time = 0;
    }

    applyPeriodicBoundaryB();
    applyPeriodicBoundaryE();

    int Ni = parameters.Ni + 2;
    int Nj = parameters.Nj + 2;
    int Nk = parameters.Nk + 2;

    if (pml_percent == 0.0) {
    int size_i_main[2] = { 0, parameters.Ni };
    int size_j_main[2] = { 0, parameters.Nj };
    int size_k_main[2] = { 0, parameters.Nk };
    double cur_coef = -4.0 * FDTD_const::PI * dt;

    cur_time = cParams.iterations;

    double Tx = cParams.period_x;
    double Ty = cParams.period_y;
    double Tz = cParams.period_z;

    int start_i = static_cast<int>(floor((-Tx / 4.0 - parameters.ax) / parameters.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - parameters.ay) / parameters.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - parameters.az) / parameters.dz));

    int max_i = static_cast<int>(floor((Tx / 4.0 - parameters.ax) / parameters.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - parameters.ay) / parameters.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - parameters.az) / parameters.dz));

    int size_i_cur[2] = { start_i, max_i };
    int size_j_cur[2] = { start_j, max_j };
    int size_k_cur[2] = { start_k, max_k };


    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < time; t++) {

        for (int k = start_k; k < max_k; ++k) {
            for (int j = start_j; j < max_j; ++j) {
                for (int i = start_i; i < max_i; ++i) {
                    int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
                    Jx[index] = init_func(static_cast<double>(i) * parameters.dx,
                                            static_cast<double>(j) * parameters.dy,
                                            static_cast<double>(k) * parameters.dz,
                                            static_cast<double>(t + 1) * cParams.dt);
                }
            }
        }



        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main,
                                        Ni, Nj, Nk,
                                        FDTD_const::C * dt / (2.0 * parameters.dx), FDTD_const::C * dt / (2.0 * parameters.dy),
                                        FDTD_const::C * dt / (2.0 * parameters.dz));
        applyPeriodicBoundaryB();

        ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
                                        Jx, Jx, Jx, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main, cur_coef, cur_time,
                                        Ni, Nj, Nk,
                                        FDTD_const::C * dt / parameters.dx, FDTD_const::C * dt / parameters.dy,
                                        FDTD_const::C * dt / parameters.dz);
        applyPeriodicBoundaryE();

        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main,
                                        Ni, Nj, Nk,
                                        FDTD_const::C * dt / (2.0 * parameters.dx), FDTD_const::C * dt / (2.0 * parameters.dy),
                                        FDTD_const::C * dt / (2.0 * parameters.dz));
        applyPeriodicBoundaryB();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "BETTER Execution time: " << elapsed.count() << " s" << std::endl;

    return;
}

}

