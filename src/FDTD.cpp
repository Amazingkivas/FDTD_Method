#include "FDTD.h"

Field::Field(const int _Ni = 1, const int _Nj = 1, const int _Nk = 1) : Ni(_Ni), Nj(_Nj), Nk(_Nk)
{
    int size = Ni * Nj * Nk;
    field = std::vector<double>(size, 0.0);
}

Field& Field::operator= (const Field& other)
{
    if (this != &other)
    {
        field = other.field;
        Ni = other.Ni;
        Nj = other.Nj;
        Nk = other.Nk;
    }
    return *this;
}

double& Field::operator() (int i, int j, int k)
{
    int i_isMinusOne = (i == -1);
    int j_isMinusOne = (j == -1);
    int k_isMinusOne = (k == -1);
    int i_isNi = (i == Ni);
    int j_isNj = (j == Nj);
    int k_isNk = (k == Nk);

    int truly_i = (Ni - 1) * i_isMinusOne + i * !(i_isMinusOne || i_isNi);
    int truly_j = (Nj - 1) * j_isMinusOne + j * !(j_isMinusOne || j_isNj);
    int truly_k = (Nk - 1) * k_isMinusOne + k * !(k_isMinusOne || k_isNk);

    int index = truly_i + truly_j * Ni + truly_k * Ni * Nj;
    return field[index];
}


FDTD::FDTD(Parameters _parameters, double _dt) : parameters(_parameters), dt(_dt)
{
    Ex = Ey = Ez = Bx = By = Bz = Field(parameters.Ni, parameters.Nj, parameters.Nk);
}

Field& FDTD::get_field(Component this_field)
{
    switch (this_field)
    {
    case Component::EX: return Ex;

    case Component::EY: return Ey;

    case Component::EZ: return Ez;

    case Component::BX: return Bx;

    case Component::BY: return By;

    case Component::BZ: return Bz;
    }
}

void FDTD::shifted_update_field(const int time)
{
    double dx = parameters.dx;
    double dy = parameters.dy;
    double dz = parameters.dz;

    for (double t = 0; t < time; t++)
    {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < parameters.Ni; i++)
        {
            for (int j = 0; j < parameters.Nj; j++)
            {
                for (int k = 0; k < parameters.Nk; k++)
                {
                    Bx(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ey(i, j, k + 1) - Ey(i, j, k)) / dz - (Ez(i, j + 1, k) - Ez(i, j, k)) / dy);
                    By(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ez(i + 1, j, k) - Ez(i, j, k)) / dx - (Ex(i, j, k + 1) - Ex(i, j, k)) / dz);
                    Bz(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ex(i, j + 1, k) - Ex(i, j, k)) / dy - (Ey(i + 1, j, k) - Ey(i, j, k)) / dx);
                }
            }
        }

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < parameters.Ni; i++)
        {
            for (int j = 0; j < parameters.Nj; j++)
            {
                for (int k = 0; k < parameters.Nk; k++)
                {
                    Ex(i, j, k) += FDTDconst::C * dt * ((Bz(i, j, k) - Bz(i, j - 1, k)) / dy - (By(i, j, k) - By(i, j, k - 1)) / dz);
                    Ey(i, j, k) += FDTDconst::C * dt * ((Bx(i, j, k) - Bx(i, j, k - 1)) / dz - (Bz(i, j, k) - Bz(i - 1, j, k)) / dx);
                    Ez(i, j, k) += FDTDconst::C * dt * ((By(i, j, k) - By(i - 1, j, k)) / dx - (Bx(i, j, k) - Bx(i, j - 1, k)) / dy);
                }
            }
        }

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < parameters.Ni; i++)
        {
            for (int j = 0; j < parameters.Nj; j++)
            {
                for (int k = 0; k < parameters.Nk; k++)
                {
                    Bx(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ey(i, j, k + 1) - Ey(i, j, k)) / dz - (Ez(i, j + 1, k) - Ez(i, j, k)) / dy);
                    By(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ez(i + 1, j, k) - Ez(i, j, k)) / dx - (Ex(i, j, k + 1) - Ex(i, j, k)) / dz);
                    Bz(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ex(i, j + 1, k) - Ex(i, j, k)) / dy - (Ey(i + 1, j, k) - Ey(i, j, k)) / dx);
                }
            }
        }
    }
}

//void FDTD::update_field(const int time)
//{
//    for (double t = 0; t < time; ++t)
//    {
//        #pragma omp parallel for collapse(2)
//        for (int j = 0; j < Nj; ++j)
//        {
//            for (int i = 0; i < Ni; ++i)
//            {
//                for (int k = 0; k < Nk; k++)
//                {
//                    Ex(i, j, k) += FDTD_Const::C * dt * (Bz(i, j + 1) - Bz(i, j - 1)) / (2.0 * dy);
//                    Ey(i, j, k) -= FDTD_Const::C * dt * (Bz(i + 1, j) - Bz(i - 1, j)) / (2.0 * dx);
//                    Ez(i, j, k) += FDTD_Const::C * dt * ((By(i + 1, j) - By(i - 1, j)) / (2.0 * dx) - (Bx(i, j + 1) - Bx(i, j - 1)) / (2.0 * dy));
//                }
//            }
//        }
//
//        #pragma omp parallel for collapse(2)
//        for (int j = 0; j < Nj; ++j)
//        {
//            for (int i = 0; i < Ni; ++i)
//            {
//                for (int k = 0; k < Nk; k++)
//                {
//                    Bx(i, j, k) -= FDTD_Const::C * dt * (Ez(i, j + 1) - Ez(i, j - 1)) / (2.0 * dy);
//                    By(i, j, k) += FDTD_Const::C * dt * (Ez(i + 1, j) - Ez(i - 1, j)) / (2.0 * dx);
//                    Bz(i, j, k) -= FDTD_Const::C * dt * ((Ey(i + 1, j) - Ey(i - 1, j)) / (2.0 * dx) - (Ex(i, j + 1) - Ex(i, j - 1)) / (2.0 * dy));
//                }
//            }
//        }
//    }
//}
