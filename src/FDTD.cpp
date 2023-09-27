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

void Borders::neighborhood(int _i, int _j)
{
    if (_i - 1 < 0)
    {
        I_prev = border_i - 1;
        I_next = _i + 1;
    }
    else if (_i + 1 >= border_i)
    {
        I_next = 0;
        I_prev = _i - 1;
    }
    else
    {
        I_next = _i + 1;
        I_prev = _i - 1;
    }

    if (_j + 1 >= border_j)
    {
        J_prev = 0;
        J_next = _j - 1;

    }
    else if (_j - 1 < 0)
    {
        J_prev = _j + 1;
        J_next = border_j - 1;
    }
    else
    {
        J_next = _j - 1;
        J_prev = _j + 1;
    }
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
    dx = (bx - ax) / Ni;
    dy = (by - ay) / Nj;
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

void FDTD::update_field(const double& time)
{
    Borders bord(Ni, Nj);

    for (double t = 0; t < time; t += dt)
    {
        for (int j = Nj - 1; j >= 0; --j)
        {
            for (int i = 0; i < Ni; ++i)
            {
                bord.neighborhood(i, j);

                Ex(i, j) += FDTD_Const::C * dt * (Bz(i, bord.j_next()) - Bz(i, bord.j_prev())) / (2.0 * dy);

                Ey(i, j) -= FDTD_Const::C * dt * (Bz(bord.i_next(), j) - Bz(bord.i_prev(), j)) / (2.0 * dx);

                Ez(i, j) += FDTD_Const::C * dt * ((By(bord.i_next(), j) - By(bord.i_prev(), j)) / (2.0 * dx) - (Bx(i, bord.j_next()) - Bx(i, bord.j_prev())) / (2.0 * dy));
            }
        }
        for (int j = Nj - 1; j >= 0; --j)
        {
            for (int i = 0; i < Ni; ++i)
            {
                bord.neighborhood(i, j);

                Bx(i, j) -= FDTD_Const::C * dt * (Ez(i, bord.j_next()) - Ez(i, bord.j_prev())) / (2.0 * dy);

                By(i, j) += FDTD_Const::C * dt * (Ez(bord.i_next(), j) - Ez(bord.i_prev(), j)) / (2.0 * dx);

                Bz(i, j) -= FDTD_Const::C * dt * ((Ey(bord.i_next(), j) - Ey(bord.i_prev(), j)) / (2.0 * dx) - (Ex(i, bord.j_next()) - Ex(i, bord.j_prev())) / (2.0 * dy));
            }
        }
    }
}
