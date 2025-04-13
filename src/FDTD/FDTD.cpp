#include "FDTD.h"

FDTD::FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar,
    std::function<double(double, double, double, double)> init_function)
    : parameters(_parameters), cParams(_Cpar), dt(_dt), pml_percent(_pml_percent), time(time_), init_func(init_function) {
    if (parameters.Ni <= 0 || parameters.Nj <= 0 || parameters.Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    int size = (parameters.Nk + 2) * (parameters.Nj + 2) * (parameters.Ni + 2);

    Jx = TimeField(time, std::vector<double, AlignedNUMA_Allocator<double>>(size, 0.0));
    //Jy = TimeField(time, std::vector<double>(size, 0.0));
    //Jz = TimeField(time, std::vector<double>(size, 0.0));

    Ex = Field(size, 0.0);
    Ey = Field(size, 0.0);
    Ez = Field(size, 0.0);
    Bx = Field(size, 0.0);
    By = Field(size, 0.0);
    Bz = Field(size, 0.0);

    // Инициализация остальных полей аналогично...

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

void FDTD::update_B_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]) {
    ComputeB_PML_FieldFunctor::apply(Ex, Ey, Ez, Bxy, Byx, Bzy, Byz, Bzx, Bxz,
                                Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
                                dt, parameters.dx, parameters.dy, parameters.dz,
                                bounds_i, bounds_j, bounds_k,
                                parameters.Ni, parameters.Nj, parameters.Nk);
}

void FDTD::update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2]) {
    ComputeE_PML_FieldFunctor::apply(Ex, Ey, Ez, Exy, Eyx, Ezy, Eyz, Ezx, Exz,
                                Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
                                dt, parameters.dx, parameters.dy, parameters.dz,
                                bounds_i, bounds_j, bounds_k,
                                parameters.Ni, parameters.Nj, parameters.Nk);
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

    InitializeCurrentFunctor::apply(Jx, cParams, parameters, init_func, std::min(cur_time, time),
                                    size_i_cur, size_j_cur, size_k_cur);
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

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < time; t++) {
        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main,
                                        Ni, Nj, Nk,
                                        FDTD_const::C * dt / (2.0 * parameters.dx), FDTD_const::C * dt / (2.0 * parameters.dy),
                                        FDTD_const::C * dt / (2.0 * parameters.dz));
        applyPeriodicBoundaryB();

        ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
                                        Jx[t], Jx[t], Jx[t], dt,
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

        //if (write_result) {
        //    std::vector<Field> return_data{ Ex, Ey, Ez, Bx, By, Bz };
        //    write_spherical(return_data, write_axis, base_path, t);
        //}
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "BETTER Execution time: " << elapsed.count() << " s" << std::endl;

    return;
}

// Остальная часть кода с PML аналогично адаптируется...
}

void FDTD::write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it) {
    std::ofstream test_fout;
    switch (axis) {
        case Axis::X: {
            for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c) {
                test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
                Field& field = fields[c];
                for (int k = 0; k < parameters.Nk; ++k) {
                    for (int j = 0; j < parameters.Nj; ++j) {
                        int index = parameters.Ni / 2 + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
                        test_fout << field[index];
                        if (j == parameters.Nj - 1) {
                            test_fout << std::endl;
                        } else {
                            test_fout << ";";
                        }
                    }
                }
                test_fout.close();
            }
            break;
        }
        case Axis::Y: {
            for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c) {
                test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
                Field& field = fields[c];
                for (int i = 0; i < parameters.Ni; ++i) {
                    for (int k = 0; k < parameters.Nk; ++k) {
                        int index = i + (parameters.Nj / 2) * parameters.Ni + k * parameters.Ni * parameters.Nj;
                        test_fout << field[index];
                        if (k == parameters.Nk - 1) {
                            test_fout << std::endl;
                        } else {
                            test_fout << ";";
                        }
                    }
                }
                test_fout.close();
            }
            break;
        }
        case Axis::Z: {
            for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c) {
                test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
                Field& field = fields[c];
                for (int i = 0; i < parameters.Ni; ++i) {
                    for (int j = 0; j < parameters.Nj; ++j) {
                        int index = i + j * parameters.Ni + (parameters.Nk / 2) * parameters.Ni * parameters.Nj;
                        test_fout << field[index];
                        if (j == parameters.Nj - 1) {
                            test_fout << std::endl;
                        } else {
                            test_fout << ";";
                        }
                    }
                }
                test_fout.close();
            }
            break;
        }
        default: throw std::logic_error("ERROR: Invalid axis");
    }
}
