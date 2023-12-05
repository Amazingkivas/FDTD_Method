#include "FDTD.h"

Field::Field(const int _Ni = 1, const int _Nj = 1) : Ni(_Ni), Nj(_Nj)
{
    int size = Ni * Nj;
    field = std::vector<double>(size, 0.0);
}

Field& Field::operator= (const Field& other)
{
    if (this != &other)
    {
        field = other.field;
        Ni = other.Ni;
        Nj = other.Nj;
    }
    return *this;
}

double& Field::operator() (int i, int j)
{
    int i_isMinusOne = (i == -1);
    int j_isMinusOne = (j == -1);
    int i_isNi = (i == Ni);
    int j_isNj = (j == Nj);
    int truly_i = (Ni - 1) * i_isMinusOne + i * !(i_isMinusOne || i_isNi);
    int truly_j = (Nj - 1) * j_isMinusOne + j * !(j_isMinusOne || j_isNj);

    int index = truly_j + truly_i * Nj;
    return field[index];
}


FDTD::FDTD(int size_grid[2], double size_x[2], double size_y[2], double _dt) : dt(_dt)
{
    Ni = size_grid[0];
    Nj = size_grid[1];
    Ex = Ey = Ez = Bx = By = Bz = Field(Ni, Nj);
    ax = size_x[0];
    bx = size_x[1];
    ay = size_y[0];
    by = size_y[1];
    dx = (bx - ax) / static_cast<double>(Ni);
    dy = (by - ay) / static_cast<double>(Nj);
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

void FDTD::update_field(const int time)
{
    for (double t = 0; t <= time; ++t)
    {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < Nj; ++j)
        {
            for (int i = 0; i < Ni; ++i)
            {
                Ex(i, j) += FDTD_Const::C * dt * (Bz(i, j + 1) - Bz(i, j - 1)) / (2.0 * dy);
                Ey(i, j) -= FDTD_Const::C * dt * (Bz(i + 1, j) - Bz(i - 1, j)) / (2.0 * dx);
                Ez(i, j) += FDTD_Const::C * dt * ((By(i + 1, j) - By(i - 1, j)) / (2.0 * dx) - (Bx(i, j + 1) - Bx(i, j - 1)) / (2.0 * dy));
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < Nj; ++j)
        {
            for (int i = 0; i < Ni; ++i)
            {
                Bx(i, j) -= FDTD_Const::C * dt * (Ez(i, j + 1) - Ez(i, j - 1)) / (2.0 * dy);
                By(i, j) += FDTD_Const::C * dt * (Ez(i + 1, j) - Ez(i - 1, j)) / (2.0 * dx);
                Bz(i, j) -= FDTD_Const::C * dt * ((Ey(i + 1, j) - Ey(i - 1, j)) / (2.0 * dx) - (Ex(i, j + 1) - Ex(i, j - 1)) / (2.0 * dy));
            }
        }
    }
}

void FDTD::shifted_update_field(const int time)
{
    for (double t = 0; t < time; t++)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < Ni; i++)
        {
            for (int j = 0; j < Nj; j++)
            {
                Bx(i, j) -= FDTD_Const::C * dt / 2.0 * (Ez(i, j + 1) - Ez(i, j)) / dy;
                By(i, j) += FDTD_Const::C * dt / 2.0 * (Ez(i + 1, j) - Ez(i, j)) / dx;
                Bz(i, j) -= FDTD_Const::C * dt / 2.0 * ((Ey(i + 1, j) - Ey(i, j)) / dx - (Ex(i, j + 1) - Ex(i, j)) / dy);
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < Ni; i++)
        {
            for (int j = 0; j < Nj; j++)
            {
                Ex(i, j) += FDTD_Const::C * dt * (Bz(i, j) - Bz(i, j - 1)) / dy;
                Ey(i, j) -= FDTD_Const::C * dt * (Bz(i, j) - Bz(i - 1, j)) / dx;
                Ez(i, j) += FDTD_Const::C * dt * ((By(i, j) - By(i - 1, j)) / dx - (Bx(i, j) - Bx(i, j - 1)) / dy);
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < Ni; i++)
        {
            for (int j = 0; j < Nj; j++)
            {
                Bx(i, j) -= FDTD_Const::C * dt / 2.0 * (Ez(i, j + 1) - Ez(i, j)) / dy;
                By(i, j) += FDTD_Const::C * dt / 2.0 * (Ez(i + 1, j) - Ez(i, j)) / dx;
                Bz(i, j) -= FDTD_Const::C * dt / 2.0 * ((Ey(i + 1, j) - Ey(i, j)) / dx - (Ex(i, j + 1) - Ex(i, j)) / dy);
            }
        }
    }
}
