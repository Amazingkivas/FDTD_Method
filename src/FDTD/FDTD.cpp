#include "FDTD.h"

FDTD::FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar,
    std::function<double(double, double, double, double)> init_function)
    : parameters(_parameters), cParams(_Cpar), dt(_dt), pml_percent(_pml_percent), time(time_), init_func(init_function) {
    if (parameters.Ni <= 0 || parameters.Nj <= 0 || parameters.Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    Jx = TimeField(time, std::vector<std::vector<std::vector<double>>>(
    parameters.Ni, std::vector<std::vector<double>>(
        parameters.Nj, std::vector<double>(parameters.Nk, 0.0))));
    Jy = TimeField(time, std::vector<std::vector<std::vector<double>>>(
    parameters.Ni, std::vector<std::vector<double>>(
        parameters.Nj, std::vector<double>(parameters.Nk, 0.0))));
    Jz = TimeField(time, std::vector<std::vector<std::vector<double>>>(
    parameters.Ni, std::vector<std::vector<double>>(
        parameters.Nj, std::vector<double>(parameters.Nk, 0.0))));

    Ex = Field(parameters.Ni, std::vector<std::vector<double>>(
    parameters.Nj, std::vector<double>(parameters.Nk, 0.0)));
    Ey = Field(parameters.Ni, std::vector<std::vector<double>>(
    parameters.Nj, std::vector<double>(parameters.Nk, 0.0)));
    Ez = Field(parameters.Ni, std::vector<std::vector<double>>(
    parameters.Nj, std::vector<double>(parameters.Nk, 0.0)));
    Bx = Field(parameters.Ni, std::vector<std::vector<double>>(
    parameters.Nj, std::vector<double>(parameters.Nk, 0.0)));
    By = Field(parameters.Ni, std::vector<std::vector<double>>(
    parameters.Nj, std::vector<double>(parameters.Nk, 0.0)));
    Bz = Field(parameters.Ni, std::vector<std::vector<double>>(
    parameters.Nj, std::vector<double>(parameters.Nk, 0.0)));

    // Инициализация остальных полей аналогично...

    pml_size_i = static_cast<int>(static_cast<double>(parameters.Ni) * pml_percent);
    pml_size_j = static_cast<int>(static_cast<double>(parameters.Nj) * pml_percent);
    pml_size_k = static_cast<int>(static_cast<double>(parameters.Nk) * pml_percent);
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

    InitializeCurrentFunctor::apply(Jx, cParams, parameters, init_func, cur_time,
                                    size_i_cur, size_j_cur, size_k_cur);
    InitializeCurrentFunctor::apply(Jy, cParams, parameters, init_func, cur_time,
                                    size_i_cur, size_j_cur, size_k_cur);
    InitializeCurrentFunctor::apply(Jz, cParams, parameters, init_func, cur_time,
                                    size_i_cur, size_j_cur, size_k_cur);
    } else {
    cur_time = 0;
    }

    if (pml_percent == 0.0) {
    int size_i_main[2] = { 0, parameters.Ni };
    int size_j_main[2] = { 0, parameters.Nj };
    int size_k_main[2] = { 0, parameters.Nk };
    for (int t = 0; t < time; t++) {
        std::cout << "Iteration: " << t + 1 << std::endl;

        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main,
                                        parameters.Ni, parameters.Nj, parameters.Nk);

        ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
                                        Jx, Jy, Jz, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main, t, cur_time,
                                        parameters.Ni, parameters.Nj, parameters.Nk);

        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
                                        parameters.dx, parameters.dy, parameters.dz,
                                        size_i_main, size_j_main, size_k_main,
                                        parameters.Ni, parameters.Nj, parameters.Nk);

        if (write_result) {
            std::vector<Field> return_data{ Ex, Ey, Ez, Bx, By, Bz };
            write_spherical(return_data, write_axis, base_path, t);
        }
    }
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
                        test_fout << field[parameters.Ni / 2][j][k];
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
                        test_fout << field[i][parameters.Nj / 2][k];
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
                        test_fout << field[i][j][parameters.Nk / 2];
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