#include <iostream>

#include "shared.h"
#include "Structures.h"
#include "FDTD_boundaries.h"

using namespace FDTD_openmp;
using namespace FDTD_struct;
using namespace FDTD_boundaries;

using std::cout;
using std::endl;

class InitializeCurrentFunctor {
public:
    InitializeCurrentFunctor(TimeField& J, CurrentParameters cParams, Parameters parameters,
                                std::function<double(double, double, double, double)> init_function)
        : J(J), cParams(cParams), parameters(parameters), init_function(init_function) {}

    static void apply(TimeField& J, CurrentParameters cParams, Parameters parameters,
                        std::function<double(double, double, double, double)> init_function,
                        int iterations, int bounds_i[2], int bounds_j[2], int bounds_k[2]) {
        InitializeCurrentFunctor functor(J, cParams, parameters, init_function);

        #pragma omp parallel for collapse(4) schedule(static)
        for (int iter = 0; iter < iterations; ++iter) {
            for (int k = bounds_k[0]; k < bounds_k[1]; ++k) {
                for (int j = bounds_j[0]; j < bounds_j[1]; ++j) {
                    for (int i = bounds_i[0]; i < bounds_i[1]; ++i) {
                        functor(iter, i, j, k);
                    }
                }
            }
        }
    }

    void operator()(const int iter, const int i, const int j, const int k) const {
        int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
        J[iter][index] = init_function(static_cast<double>(i) * parameters.dx,
                                            static_cast<double>(j) * parameters.dy,
                                            static_cast<double>(k) * parameters.dz,
                                            static_cast<double>(iter + 1) * cParams.dt);
    }

private:
    TimeField& J;
    CurrentParameters cParams;
    Parameters parameters;
    std::function<double(double, double, double, double)> init_function;
};

class ComputeE_FieldFunctor {
public:
    ComputeE_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bx, Field& By, Field& Bz,
                        Field& Jx, Field& Jy, Field& Jz,
                        double& dt, double& dx, double& dy, double& dz, double& cur_coef, int& iters,
                        int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz),
        Jx(Jx), Jy(Jy), Jz(Jz), dt(dt), dx(dx), dy(dy), dz(dz), cur_coef(cur_coef), iters(iters),
        Ni(Ni), Nj(Nj), Nk(Nk), coef_dx(coef_dx), coef_dy(coef_dy), coef_dz(coef_dz) {}

    static inline void apply(Field& Ex, Field& Ey, Field& Ez,
                    Field& Bx, Field& By, Field& Bz,
                    Field& Jx, Field& Jy, Field& Jz,
                    double& dt, double& dx, double& dy, double& dz,
                    int bounds_i[2], int bounds_j[2], int bounds_k[2], double& cur_coef, int& iters,
                    int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz) {
        //ComputeE_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dt, dx, dy, dz, cur_coef, iters, Ni, Nj, Nk, coef_dx, coef_dy, coef_dz);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int k = bounds_k[0] + 1; k <= bounds_k[1]; k++) {
            for (int j = bounds_j[0] + 1; j <= bounds_j[1]; j++) {
                int index_kj = j * Ni + k * Ni * Nj;
                int j_pred_kj = (j - 1) * Ni + k * Ni * Nj;
                int k_pred_kj = j * Ni + (k - 1) * Ni * Nj;
                #pragma omp simd
                for (int i = bounds_i[0] + 1; i <= bounds_i[1]; i++) {
                    int index = i + index_kj;
                    int i_pred = index - 1;
                    int j_pred = i + j_pred_kj;
                    int k_pred = i + k_pred_kj;

                    Ex[index] += cur_coef * Jx[index] + coef_dy * (Bz[index] - Bz[j_pred]) - coef_dz * (By[index] - By[k_pred]);
                    Ey[index] += cur_coef * Jy[index] + coef_dz * (Bx[index] - Bx[k_pred]) - coef_dx * (Bz[index] - Bz[i_pred]);
                    Ez[index] += cur_coef * Jz[index] + coef_dx * (By[index] - By[i_pred]) - coef_dy * (Bx[index] - Bx[j_pred]);
                }
            }
        }
    }

    void operator()(const int& i, const int& j, const int& k) const {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        int index = i + j * Ni + k * Ni * Nj;

        i_pred = i_pred + j * Ni + k * Ni * Nj;
        j_pred = i + j_pred * Ni + k * Ni * Nj;
        k_pred = i + j * Ni + k_pred * Ni * Nj;

        Ex[index] += cur_coef * Jx[index] + coef_dy * (Bz[index] - Bz[j_pred]) - coef_dz * (By[index] - By[k_pred]);
        Ey[index] += cur_coef * Jy[index] + coef_dz * (Bx[index] - Bx[k_pred]) - coef_dx * (Bz[index] - Bz[i_pred]);
        Ez[index] += cur_coef * Jz[index] + coef_dx * (By[index] - By[i_pred]) - coef_dy * (Bx[index] - Bx[j_pred]);

    }

private:
    double cur_coef, coef_dx, coef_dy, coef_dz;
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    Field &Jx, &Jy, &Jz;
    double dt, dx, dy, dz;
    int t, iters;
    int Ni, Nj, Nk;
};

class ComputeB_FieldFunctor {
public:
    ComputeB_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                            Field& Bx, Field& By, Field& Bz,
                            double& dt, double& dx, double& dy, double& dz,
                            int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bx(Bx), By(By), Bz(Bz), dt(dt), dx(dx), dy(dy), dz(dz),
            Ni(Ni), Nj(Nj), Nk(Nk), coef_dx(coef_dx), coef_dy(coef_dy), coef_dz(coef_dz) {}

    static inline void apply(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bx, Field& By, Field& Bz,
                        double& dt, double& dx, double& dy, double& dz,
                        int bounds_i[2], int bounds_j[2], int bounds_k[2],
                        int& Ni, int& Nj, int& Nk, double coef_dx, double coef_dy, double coef_dz) {
        //ComputeB_FieldFunctor functor(Ex, Ey, Ez, Bx, By, Bz, dt, dx, dy, dz, Ni, Nj, Nk, coef_dx, coef_dy, coef_dz);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int k = bounds_k[0] + 1; k <= bounds_k[1]; k++) {
            for (int j = bounds_j[0] + 1; j <= bounds_j[1]; j++) {
                int index_kj = j * Ni + k * Ni * Nj;
                int j_next_kj = (j + 1) * Ni + k * Ni * Nj;
                int k_next_kj = j * Ni + (k + 1) * Ni * Nj;
                #pragma omp simd
                for (int i = bounds_i[0] + 1; i <= bounds_i[1]; i++) {
                    int index = i + index_kj;
                    int i_next = index + 1;
                    int j_next = i + j_next_kj;
                    int k_next = i + k_next_kj;

                    Bx[index] += coef_dz * (Ey[k_next] - Ey[index]) - coef_dy * (Ez[j_next] - Ez[index]);
                    By[index] += coef_dx * (Ez[i_next] - Ez[index]) - coef_dz * (Ex[k_next] - Ex[index]);
                    Bz[index] += coef_dy * (Ex[j_next] - Ex[index]) - coef_dx * (Ey[i_next] - Ey[index]);
                }
            }
        }
    }

    void operator()(const int& i, const int& j, const int& k) const {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        int index = i + j * Ni + k * Ni * Nj;

        i_next = i_next + j * Ni + k * Ni * Nj;
        j_next = i + j_next * Ni + k * Ni * Nj;
        k_next = i + j * Ni + k_next * Ni * Nj;

        Bx[index] += coef_dz * (Ey[k_next] - Ey[index]) - coef_dy * (Ez[j_next] - Ez[index]);
        By[index] += coef_dx * (Ez[i_next] - Ez[index]) - coef_dz * (Ex[k_next] - Ex[index]);
        Bz[index] += coef_dy * (Ex[j_next] - Ex[index]) - coef_dx * (Ey[i_next] - Ey[index]);
    }

private:
    double cur_coef, coef_dx, coef_dy, coef_dz;
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;
};

class ComputeE_PML_FieldFunctor {
public:
    ComputeE_PML_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                                Field& Exy, Field& Eyx, Field& Ezy,
                                Field& Eyz, Field& Ezx, Field& Exz,
                                Field& Bx, Field& By, Field& Bz,
                                Field& EsigmaX, Field& EsigmaY, Field& EsigmaZ,
                                double dt, double dx, double dy, double dz,
                                int Ni, int Nj, int Nk)
        : Ex(Ex), Ey(Ey), Ez(Ez), Exy(Exy), Eyx(Eyx), Ezy(Ezy),
            Eyz(Eyz), Ezx(Ezx), Exz(Exz), Bx(Bx), By(By), Bz(Bz),
            EsigmaX(EsigmaX), EsigmaY(EsigmaY), EsigmaZ(EsigmaZ),
            dt(dt), dx(dx), dy(dy), dz(dz), Ni(Ni), Nj(Nj), Nk(Nk) {}

    static void apply(Field& Ex, Field& Ey, Field& Ez,
                        Field& Exy, Field& Eyx, Field& Ezy,
                        Field& Eyz, Field& Ezx, Field& Exz,
                        Field& Bx, Field& By, Field& Bz,
                        Field& EsigmaX, Field& EsigmaY, Field& EsigmaZ,
                        double dt, double dx, double dy, double dz,
                        int bounds_i[2], int bounds_j[2], int bounds_k[2],
                        int Ni, int Nj, int Nk) {
        ComputeE_PML_FieldFunctor functor(Ex, Ey, Ez, Exy, Eyx, Ezy, Eyz, Ezx, Exz,
                                            Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
                                            dt, dx, dy, dz, Ni, Nj, Nk);

        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = bounds_i[0]; i < bounds_i[1]; ++i) {
            for (int j = bounds_j[0]; j < bounds_j[1]; ++j) {
                for (int k = bounds_k[0]; k < bounds_k[1]; ++k) {
                    functor(i, j, k);
                }
            }
        }
    }

    void operator()(const int i, const int j, const int k) const {
        int i_pred = i - 1;
        int j_pred = j - 1;
        int k_pred = k - 1;

        FDTD_boundaries::applyPeriodicBoundary(i_pred, j_pred, k_pred, Ni, Nj, Nk);

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        /*if (EsigmaX[i][j][k] != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(EsigmaX[i][j][k])) / (EsigmaX[i][j][k] * dx);
        else
            PMLcoef2_x = FDTD_const::C * dt / dx;

        if (EsigmaY[i][j][k] != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(EsigmaY[i][j][k])) / (EsigmaY[i][j][k] * dy);
        else
            PMLcoef2_y = FDTD_const::C * dt / dy;

        if (EsigmaZ[i][j][k] != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ[i][j][k])) / (EsigmaZ[i][j][k] * dz);
        else
            PMLcoef2_z = FDTD_const::C * dt / dz;

        Eyx[i][j][k] = Eyx[i][j][k] * PMLcoef(EsigmaX[i][j][k]) -
                        PMLcoef2_x * (Bz[i][j][k] - Bz[i_pred][j][k]);
        Ezx[i][j][k] = Ezx[i][j][k] * PMLcoef(EsigmaX[i][j][k]) +
                        PMLcoef2_x * (By[i][j][k] - By[i_pred][j][k]);

        Exy[i][j][k] = Exy[i][j][k] * PMLcoef(EsigmaY[i][j][k]) +
                        PMLcoef2_y * (Bz[i][j][k] - Bz[i][j_pred][k]);
        Ezy[i][j][k] = Ezy[i][j][k] * PMLcoef(EsigmaY[i][j][k]) -
                        PMLcoef2_y * (Bx[i][j][k] - Bx[i][j_pred][k]);

        Exz[i][j][k] = Exz[i][j][k] * PMLcoef(EsigmaZ[i][j][k]) -
                        PMLcoef2_z * (By[i][j][k] - By[i][j][k_pred]);
        Eyz[i][j][k] = Eyz[i][j][k] * PMLcoef(EsigmaZ[i][j][k]) +
                        PMLcoef2_z * (Bx[i][j][k] - Bx[i][j][k_pred]);

        Ex[i][j][k] = Exz[i][j][k] + Exy[i][j][k];
        Ey[i][j][k] = Eyx[i][j][k] + Eyz[i][j][k];
        Ez[i][j][k] = Ezy[i][j][k] + Ezx[i][j][k];*/
    }

private:
    Field &Ex, &Ey, &Ez;
    Field &Exy, &Eyx, &Ezy, &Eyz, &Ezx, &Exz;
    Field &Bx, &By, &Bz;
    Field &EsigmaX, &EsigmaY, &EsigmaZ;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;

    double PMLcoef(double sigma) const {
        return std::exp(-sigma * dt * FDTD_const::C);
    }
};

class ComputeB_PML_FieldFunctor {
public:
    ComputeB_PML_FieldFunctor(Field& Ex, Field& Ey, Field& Ez,
                                Field& Bxy, Field& Byx, Field& Bzy,
                                Field& Byz, Field& Bzx, Field& Bxz,
                                Field& Bx, Field& By, Field& Bz,
                                Field& BsigmaX, Field& BsigmaY, Field& BsigmaZ,
                                double dt, double dx, double dy, double dz,
                                int Ni, int Nj, int Nk)
        : Ex(Ex), Ey(Ey), Ez(Ez), Bxy(Bxy), Byx(Byx), Bzy(Bzy),
            Byz(Byz), Bzx(Bzx), Bxz(Bxz), Bx(Bx), By(By), Bz(Bz),
            BsigmaX(BsigmaX), BsigmaY(BsigmaY), BsigmaZ(BsigmaZ),
            dt(dt), dx(dx), dy(dy), dz(dz), Ni(Ni), Nj(Nj), Nk(Nk) {}

    static void apply(Field& Ex, Field& Ey, Field& Ez,
                        Field& Bxy, Field& Byx, Field& Bzy,
                        Field& Byz, Field& Bzx, Field& Bxz,
                        Field& Bx, Field& By, Field& Bz,
                        Field& BsigmaX, Field& BsigmaY, Field& BsigmaZ,
                        double dt, double dx, double dy, double dz,
                        int bounds_i[2], int bounds_j[2], int bounds_k[2],
                        int Ni, int Nj, int Nk) {
        ComputeB_PML_FieldFunctor functor(Ex, Ey, Ez, Bxy, Byx, Bzy, Byz, Bzx, Bxz,
                                            Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
                                            dt, dx, dy, dz, Ni, Nj, Nk);

        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = bounds_i[0]; i < bounds_i[1]; ++i) {
            for (int j = bounds_j[0]; j < bounds_j[1]; ++j) {
                for (int k = bounds_k[0]; k < bounds_k[1]; ++k) {
                    functor(i, j, k);
                }
            }
        }
    }

    void operator()(const int i, const int j, const int k) const {
        int i_next = i + 1;
        int j_next = j + 1;
        int k_next = k + 1;

        FDTD_boundaries::applyPeriodicBoundary(i_next, j_next, k_next, Ni, Nj, Nk);

        double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

        /*if (BsigmaX[i][j][k] != 0.0)
            PMLcoef2_x = (1.0 - PMLcoef(BsigmaX[i][j][k])) / (BsigmaX[i][j][k] * dx);
        else
            PMLcoef2_x = FDTD_const::C * dt / dx;

        if (BsigmaY[i][j][k] != 0.0)
            PMLcoef2_y = (1.0 - PMLcoef(BsigmaY[i][j][k])) / (BsigmaY[i][j][k] * dy);
        else
            PMLcoef2_y = FDTD_const::C * dt / dy;

        if (BsigmaZ[i][j][k] != 0.0)
            PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ[i][j][k])) / (BsigmaZ[i][j][k] * dz);
        else
            PMLcoef2_z = FDTD_const::C * dt / dz;

        Byx[i][j][k] = Byx[i][j][k] * PMLcoef(BsigmaX[i][j][k]) +
                        PMLcoef2_x * (Ez[i_next][j][k] - Ez[i][j][k]);
        Bzx[i][j][k] = Bzx[i][j][k] * PMLcoef(BsigmaX[i][j][k]) -
                        PMLcoef2_x * (Ey[i_next][j][k] - Ey[i][j][k]);

        Bxy[i][j][k] = Bxy[i][j][k] * PMLcoef(BsigmaY[i][j][k]) -
                        PMLcoef2_y * (Ez[i][j_next][k] - Ez[i][j][k]);
        Bzy[i][j][k] = Bzy[i][j][k] * PMLcoef(BsigmaY[i][j][k]) +
                        PMLcoef2_y * (Ex[i][j_next][k] - Ex[i][j][k]);

        Bxz[i][j][k] = Bxz[i][j][k] * PMLcoef(BsigmaZ[i][j][k]) +
                        PMLcoef2_z * (Ey[i][j][k_next] - Ey[i][j][k]);
        Byz[i][j][k] = Byz[i][j][k] * PMLcoef(BsigmaZ[i][j][k]) -
                        PMLcoef2_z * (Ex[i][j][k_next] - Ex[i][j][k]);

        Bx[i][j][k] = Bxy[i][j][k] + Bxz[i][j][k];
        By[i][j][k] = Byz[i][j][k] + Byx[i][j][k];
        Bz[i][j][k] = Bzx[i][j][k] + Bzy[i][j][k];*/
    }

private:
    Field &Ex, &Ey, &Ez;
    Field &Bxy, &Byx, &Bzy, &Byz, &Bzx, &Bxz;
    Field &Bx, &By, &Bz;
    Field &BsigmaX, &BsigmaY, &BsigmaZ;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;

    double PMLcoef(double sigma) const {
        return std::exp(-sigma * dt * FDTD_const::C);
    }
};


