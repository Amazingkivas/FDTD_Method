#include <iostream>

#include "shared.h"
#include "Structures.h"

using namespace FDTD_openmp;
using namespace FDTD_struct;

using std::cout;
using std::endl;

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

private:
    double cur_coef, coef_dx, coef_dy, coef_dz;
    Field &Ex, &Ey, &Ez;
    Field &Bx, &By, &Bz;
    double dt, dx, dy, dz;
    int Ni, Nj, Nk;
};
