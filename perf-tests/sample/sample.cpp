#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <string>

#include "test_FDTD.h"
#include "FDTD_PML.h"

using namespace FDTD_openmp;


void spherical_wave(int n, int it, std::string base_path = "") {
    CurrentParameters cur_param {
        8,
        4,
        0.2,
    };
    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;
    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);
    std::function<double(double, double, double, double)> cur_func 
        = [T, Tx, Ty, Tz](double x, double y, double z, double t) {
        return sin(2.0 * FDTD_const::PI * t / T) 
            * pow(cos(2.0 * FDTD_const::PI * x / Tx), 2.0) 
            * pow(cos(2.0 * FDTD_const::PI * y / Ty), 2.0) 
            * pow(cos(2.0 * FDTD_const::PI * z / Tz), 2.0);
    };
    
    // Initialization of the structures and method
    double d = FDTD_const::C;

    double boundary = static_cast<double>(n) / 2.0 * d;

    Parameters params {
        n,          // Ni
        n,          // Nj
        n,          // Nk
        -boundary,  // x_min
        boundary,   // x_max
        -boundary,  // y_min
        boundary,   // y_max
        -boundary,  // z_min
        boundary,   // z_max
        d,          // dx
        d,          // dy
        d           // dz
    };

    FDTD_openmp::FDTD method(params, cur_param.dt);

    int cur_time = std::min(cur_param.iterations, it);

    int start_i = static_cast<int>(floor((-Tx / 4.0 - params.ax) / params.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - params.ay) / params.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - params.az) / params.dz));

    int max_i = static_cast<int>(floor((Tx / 4.0 - params.ax) / params.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - params.ay) / params.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - params.az) / params.dz));

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < cur_time; t++) {
        for (int k = start_k; k < max_k; ++k) {
            for (int j = start_j; j < max_j; ++j) {
                for (int i = start_i; i < max_i; ++i) {
                    int index = i + j * params.Ni + k * params.Ni * params.Nj;
                    double value = cur_func(static_cast<double>(i) * params.dx,
                                            static_cast<double>(j) * params.dy,
                                            static_cast<double>(k) * params.dz,
                                            static_cast<double>(t + 1) * cur_param.dt);

                    method.get_field(Component::JX)[index] = value;
                    method.get_field(Component::JY)[index] = value;
                    method.get_field(Component::JZ)[index] = value;
                }
            }
        }
        method.update_fields();
    }
    method.zeroed_currents();
    for (int t = cur_time; t < it; t++) {
        method.update_fields();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;

#ifdef __PML_TEST__
    FDTD_openmp::FDTD_PML pml_method(params, cur_param.dt, 0.2);

    auto start_pml = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < cur_time; t++) {
        for (int k = start_k; k < max_k; ++k) {
            for (int j = start_j; j < max_j; ++j) {
                for (int i = start_i; i < max_i; ++i) {
                    int index = i + j * params.Ni + k * params.Ni * params.Nj;
                    double value = cur_func(static_cast<double>(i) * params.dx,
                                            static_cast<double>(j) * params.dy,
                                            static_cast<double>(k) * params.dz,
                                            static_cast<double>(t + 1) * cur_param.dt);

                    pml_method.get_field(Component::JX)[index] = value;
                    pml_method.get_field(Component::JY)[index] = value;
                    pml_method.get_field(Component::JZ)[index] = value;
                }
            }
        }
        pml_method.update_fields();
    }
    pml_method.zeroed_currents();
    for (int t = cur_time; t < it; t++) {
        pml_method.update_fields();
    }
    auto end_pml = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_pml = end_pml - start_pml;
    std::cout << "Execution time (PML): " << elapsed_pml.count() << " s" << std::endl;
#endif //__PML_TEST__

    int k = params.Nk/2;
    for (int j = params.Nj/2 - 5; j < params.Nj/2 + 5; j++) {
        for (int i = params.Ni/2 - 5; i < params.Ni/2 + 5; i++) {
            int index = i + j * params.Ni + k * params.Ni * params.Nj;
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) 
                  << method.get_field(Component::EX)[index];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

#ifdef __PML_TEST__
    std::cout << "PML: \n" << std::endl;
    for (int j = params.Nj/2 - 5; j < params.Nj/2 + 5; j++) {
        for (int i = params.Ni/2 - 5; i < params.Ni/2 + 5; i++) {
            int index = i + j * params.Ni + k * params.Ni * params.Nj;
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) 
                  << pml_method.get_field(Component::EX)[index];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif //__PML_TEST__
}

// Deposits a point current located between grid nodes with cloud-in-cell (CIC)
// interpolation.  The triangular CIC weights distribute the source over the
// four adjacent cells in the x-y plane and sum to one.
void interpolated_point_source(int n, int it) {
    if (n < 2) {
        throw std::invalid_argument("CIC interpolation requires at least two cells per axis");
    }

    CurrentParameters cur_param {
        8,
        4,
        0.2,
    };

    double d = FDTD_const::C;
    double boundary = static_cast<double>(n) / 2.0 * d;
    Parameters params {
        n, n, n,
        -boundary, boundary,
        -boundary, boundary,
        -boundary, boundary,
        d, d, d
    };

    FDTD_openmp::FDTD method(params, cur_param.dt);
    int source_time = std::min(
        static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt), it);

    // An off-node position makes all four CIC weights non-zero.
    double source_x = params.ax +
        (static_cast<double>(n - 2) / 2.0 + 0.35) * params.dx;
    double source_y = params.ay +
        (static_cast<double>(n - 2) / 2.0 + 0.65) * params.dy;
    double grid_x = (source_x - params.ax) / params.dx;
    double grid_y = (source_y - params.ay) / params.dy;
    int i0 = static_cast<int>(std::floor(grid_x));
    int j0 = static_cast<int>(std::floor(grid_y));
    double wx1 = grid_x - static_cast<double>(i0);
    double wy1 = grid_y - static_cast<double>(j0);
    double wx0 = 1.0 - wx1;
    double wy0 = 1.0 - wy1;
    int k = params.Nk / 2;

    auto index = [&params, k](int i, int j) {
        return i + j * params.Ni + k * params.Ni * params.Nj;
    };

    std::cout << "Interpolation mode: CIC point source with weights "
              << wx0 * wy0 << ", " << wx1 * wy0 << ", "
              << wx0 * wy1 << ", " << wx1 * wy1 << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < it; ++t) {
        method.zeroed_currents();
        if (t < source_time) {
            double value = std::sin(2.0 * FDTD_const::PI *
                                    static_cast<double>(t + 1) * cur_param.dt /
                                    static_cast<double>(cur_param.period));
            Field& current = method.get_field(Component::JX);
            current[index(i0, j0)] += value * wx0 * wy0;
            current[index(i0 + 1, j0)] += value * wx1 * wy0;
            current[index(i0, j0 + 1)] += value * wx0 * wy1;
            current[index(i0 + 1, j0 + 1)] += value * wx1 * wy1;
        }
        method.update_fields();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time (CIC interpolation): " << elapsed.count() << " s"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::ifstream source_fin;
    std::vector<char*> arguments(argv, argv + argc);
    const char* sample_mode = std::getenv("FDTD_SAMPLE_MODE");
    bool use_interpolation = sample_mode != nullptr &&
        std::string(sample_mode) == "interpolation";

    if (argc == 1) {
        int N = 32;
        int Iterations = 100;
        if (use_interpolation) {
            interpolated_point_source(N, Iterations);
        } else {
            spherical_wave(N, Iterations, "../../");
        }
    }
    else if (argc == 3) {
        int N = std::atoi(arguments[1]);
        int Iterations = std::atoi(arguments[2]);
        if (use_interpolation) {
            interpolated_point_source(N, Iterations);
        } else {
            spherical_wave(N, Iterations);
        }
    }
    else {
        std::cout << "ERROR: Incorrect number of parameters" << std::endl;
        exit(1);
    }
    return 0;
}
