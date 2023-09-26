#include "FDTD.h"

Field::Field(const int _Ni = 1, const int _Nj = 1) : Ni(_Ni), Nj(_Nj)
{
    field = std::vector<double>(Ni * Nj, 0.0);
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
    case EX: return Ex;

    case EY: return Ey;

    case EZ: return Ez;

    case BX: return Bx;

    case BY: return By;

    case BZ: return Bz;
    }
}

void FDTD::update_field(const double& time)
{
    int I_next = 0;
    int J_next = 0;
    int I_prev = 0;
    int J_prev = 0;

    for (double t = 0; t < time; t += dt)
    {
        for (int j = Nj - 1; j >= 0; --j)
        {
            for (int i = 0; i < Ni; ++i)
            {
                if (i - 1 < 0)
                {
                    I_prev = Ni - 1;
                    I_next = i + 1;
                }
                else if (i + 1 >= Ni)
                {
                    I_next = 0;
                    I_prev = i - 1;
                }
                else
                {
                    I_next = i + 1;
                    I_prev = i - 1;
                }
                if (j + 1 >= Nj)
                {
                    J_prev = 0;
                    J_next = j - 1;

                }
                else if (j - 1 < 0)
                {
                    J_prev = j + 1;
                    J_next = Nj - 1;
                }
                else
                {
                    J_next = j - 1;
                    J_prev = j + 1;
                }

                Ex(i, j) += C * dt * (Bz(i, J_next) - Bz(i, J_prev)) / (2 * dy);

                Ey(i, j) -= C * dt * (Bz(I_next, j) - Bz(I_prev, j)) / (2 * dx);

                Ez(i, j) += C * dt * ((By(I_next, j) - By(I_prev, j)) / (2 * dx) - (Bx(i, J_next) - Bx(i, J_prev)) / (2 * dy));
            }
        }
    
        for (int j = Nj - 1; j >= 0; --j)
        {
            for (int i = 0; i < Ni; ++i)
            {
                if (i - 1 < 0)
                {
                    I_prev = Ni - 1;
                    I_next = i + 1;
                }
                else if (i + 1 >= Ni)
                {
                    I_next = 0;
                    I_prev = i - 1;
                }
                else
                {
                    I_next = i + 1;
                    I_prev = i - 1;
                }
                if (j + 1 >= Nj)
                {
                    J_prev = 0;
                    J_next = j - 1;

                }
                else if (j - 1 < 0)
                {
                    J_prev = j + 1;
                    J_next = Nj - 1;
                }
                else
                {
                    J_next = j - 1;
                    J_prev = j + 1;
                }

                Bx(i, j) -= C * dt * (Ez(i, J_next) - Ez(i, J_prev)) / (2 * dy);

                By(i, j) += C * dt * (Ez(I_next, j) - Ez(I_prev, j)) / (2 * dx);

                Bz(i, j) -= C * dt * ((Ey(I_next, j) - Ey(I_prev, j)) / (2 * dx) - (Ex(i, J_next) - Ex(i, J_prev)) / (2 * dy));
            }
        }
    }
}
