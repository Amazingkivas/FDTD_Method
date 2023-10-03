#include "FDTD.h"

int Cell_number::operator+ (int number) const
{
    if (current + number >= border)
    {
        return 0;
    }
    else
    {
        return current + number;
    }
}
int Cell_number::operator- (int number) const
{
    if (current - number < 0)
    {
        return border - 1;
    }
    else
    {
        return current - number;
    }
}
int Cell_number::operator* ()
{
    return current;
}
Cell_number& Cell_number::operator++ ()
{
    ++current;
    if (current > border)
    {
        current = 0;
    }
    return *this;
}
bool Cell_number::operator< (int other)
{
    return current < other;
}

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

double& Field::operator() (int _i, int _j)
{
    int index = _j + _i * Nj;
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

void FDTD::update_field(const double& time)
{
    for (double t = 0; t < time; t += dt)
    {
        for (Cell_number j(Nj); j < Nj; ++j)
        {
            for (Cell_number i(Ni); i < Ni; ++i)
            {
                Ex(*i, *j) += FDTD_Const::C * dt * (Bz(*i, j + 1) - Bz(*i, j - 1)) / (2.0 * dy);

                Ey(*i, *j) -= FDTD_Const::C * dt * (Bz(i + 1, *j) - Bz(i - 1, *j)) / (2.0 * dx);

                Ez(*i, *j) += FDTD_Const::C * dt * ((By(i + 1, *j) - By(i - 1, *j)) / (2.0 * dx) - (Bx(*i, j + 1) - Bx(*i, j - 1)) / (2.0 * dy));
            }
        }
        for (Cell_number j(Nj); j < Nj; ++j)
        {
            for (Cell_number i(Ni); i < Ni; ++i)
            {
                Bx(*i, *j) -= FDTD_Const::C * dt * (Ez(*i, j + 1) - Ez(*i, j - 1)) / (2.0 * dy);

                By(*i, *j) += FDTD_Const::C * dt * (Ez(i + 1, *j) - Ez(i - 1, *j)) / (2.0 * dx);

                Bz(*i, *j) -= FDTD_Const::C * dt * ((Ey(i + 1, *j) - Ey(i - 1, *j)) / (2.0 * dx) - (Ex(*i, j + 1) - Ex(*i, j - 1)) / (2.0 * dy));
            }
        }
    }
}
