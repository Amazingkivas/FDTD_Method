#include "FDTD.h"
#include <iostream>

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
    if (parameters.ax > parameters.bx ||
        parameters.ay > parameters.by ||
        parameters.az > parameters.bz)
    {
        throw std::exception("ERROR: invalid parameters");
    }
    if (parameters.Ni <= 0 ||
        parameters.Nj <= 0 ||
        parameters.Nk <= 0 ||
        dt <= 0)
    {
        throw std::exception("ERROR: invalid parameters");
    }
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

    default: throw std::exception("ERROR: Invalid field component");
    }
}

std::vector<Field>& FDTD::get_current(Component this_current)
{
    switch (this_current)
    {
    case Component::JX: return Jx;

    case Component::JY: return Jy;

    case Component::JZ: return Jz;

    default: throw std::exception("ERROR: Invalid current component");
    }
}

void FDTD::update_fields(const int time)
{
    if (time < 0)
    {
        throw std::exception("ERROR: Invalid update field argument");
    }
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
                    Ex(i, j, k) = Ex(i, j, k) - 4.0 * M_PI * dt * Jx[t](i, j, k) + FDTDconst::C * dt * ((Bz(i, j, k) - Bz(i, j - 1, k)) / dy - (By(i, j, k) - By(i, j, k - 1)) / dz);
                    Ey(i, j, k) = Ey(i, j, k) - 4.0 * M_PI * dt * Jy[t](i, j, k) + FDTDconst::C * dt * ((Bx(i, j, k) - Bx(i, j, k - 1)) / dz - (Bz(i, j, k) - Bz(i - 1, j, k)) / dx);
                    Ez(i, j, k) = Ez(i, j, k) - 4.0 * M_PI * dt * Jz[t](i, j, k) + FDTDconst::C * dt * ((By(i, j, k) - By(i - 1, j, k)) / dx - (Bx(i, j, k) - Bx(i, j - 1, k)) / dy);
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
