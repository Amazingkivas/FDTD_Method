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
        int j;
        int i;

        // Left boundary conditions
        Ex(0, 0) += FDTD_Const::C * dt * (Bz(0, 1) - Bz(0, Nj - 1)) / (2.0 * dy);
        Ez(0, 0) += FDTD_Const::C * dt * ((By(1, 0) - By(Ni - 1, 0)) / (2.0 * dx) - (Bx(0, 1) - Bx(0, Nj - 1)) / (2.0 * dy));
        for (i = 1; i < Ni - 1; ++i)
        {
            Ex(i, 0) += FDTD_Const::C * dt * (Bz(i, 1) - Bz(i, Nj - 1)) / (2.0 * dy);
            Ez(i, 0) += FDTD_Const::C * dt * ((By(i + 1, 0) - By(i - 1, 0)) / (2.0 * dx) - (Bx(i, 1) - Bx(i, Nj - 1)) / (2.0 * dy));
        }
        Ex(Ni - 1, 0) += FDTD_Const::C * dt * (Bz(Ni - 1, 1) - Bz(Ni - 1, Nj - 1)) / (2.0 * dy);
        Ez(Ni - 1, 0) += FDTD_Const::C * dt * ((By(0, 0) - By(Ni - 2, 0)) / (2.0 * dx) - (Bx(Ni - 1, 1) - Bx(Ni - 1, Nj - 1)) / (2.0 * dy));

        // Basic parallelized cycles
        #pragma omp parallel private (i,j) default(shared)
        {
            #pragma omp for schedule(static)
            for (j = 1; j < Nj - 1; ++j)
            {
                for (i = 0; i < Ni; ++i)
                {
                    Ex(i, j) += FDTD_Const::C * dt * (Bz(i, j + 1) - Bz(i, j - 1)) / (2.0 * dy);
                }
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj; ++j)
            {
                Ey(0, j) -= FDTD_Const::C * dt * (Bz(1, j) - Bz(Ni - 1, j)) / (2.0 * dx);
                for (i = 1; i < Ni - 1; ++i)
                {
                    Ey(i, j) -= FDTD_Const::C * dt * (Bz(i + 1, j) - Bz(i - 1, j)) / (2.0 * dx);
                }
                Ey(Ni - 1, j) -= FDTD_Const::C * dt * (Bz(0, j) - Bz(Ni - 2, j)) / (2.0 * dx);
            }
            #pragma omp for schedule(static)
            for (j = 1; j < Nj - 1; ++j)
            {
                Ez(0, j) += FDTD_Const::C * dt * ((By(1, j) - By(Ni - 1, j)) / (2.0 * dx) - (Bx(0, j + 1) - Bx(0, j - 1)) / (2.0 * dy));
                for (i = 1; i < Ni - 1; ++i)
                {
                    Ez(i, j) += FDTD_Const::C * dt * ((By(i + 1, j) - By(i - 1, j)) / (2.0 * dx) - (Bx(i, j + 1) - Bx(i, j - 1)) / (2.0 * dy));
                }
                Ez(Ni - 1, j) += FDTD_Const::C * dt * ((By(0, j) - By(Ni - 2, j)) / (2.0 * dx) - (Bx(Ni - 1, j + 1) - Bx(Ni - 1, j - 1)) / (2.0 * dy));
            }
        }
        // Right boundary conditions
        Ex(0, Nj - 1) += FDTD_Const::C * dt * (Bz(0, 0) - Bz(0, Nj - 2)) / (2.0 * dy);
        Ez(0, Nj - 1) += FDTD_Const::C * dt * ((By(1, Nj - 1) - By(Ni - 1, Nj - 1)) / (2.0 * dx) - (Bx(0, 0) - Bx(0, Nj - 2)) / (2.0 * dy));
        for (i = 1; i < Ni - 1; ++i)
        {
            Ex(i, Nj - 1) += FDTD_Const::C * dt * (Bz(i, 0) - Bz(i, Nj - 2)) / (2.0 * dy);
            Ez(i, Nj - 1) += FDTD_Const::C * dt * ((By(i + 1, Nj - 1) - By(i - 1, Nj - 1)) / (2.0 * dx) - (Bx(i, 0) - Bx(i, Nj - 2)) / (2.0 * dy));
        }
        Ex(Ni - 1, Nj - 1) += FDTD_Const::C * dt * (Bz(Ni - 1, 0) - Bz(Ni - 1, Nj - 2)) / (2.0 * dy);
        Ez(Ni - 1, Nj - 1) += FDTD_Const::C * dt * ((By(0, Nj - 1) - By(Ni - 2, Nj - 1)) / (2.0 * dx) - (Bx(Ni - 1, 0) - Bx(Ni - 1, Nj - 2)) / (2.0 * dy));


        // Left boundary conditions
        Bx(0, 0) -= FDTD_Const::C * dt * (Ez(0, 1) - Ez(0, Nj - 1)) / (2.0 * dy);
        Bz(0, 0) -= FDTD_Const::C * dt * ((Ey(1, 0) - Ey(Ni - 1, 0)) / (2.0 * dx) - (Ex(0, 1) - Ex(0, Nj - 1)) / (2.0 * dy));
        for (i = 1; i < Ni - 1; ++i)
        {
            Bx(i, 0) -= FDTD_Const::C * dt * (Ez(i, 1) - Ez(i, Nj - 1)) / (2.0 * dy);
            Bz(i, 0) -= FDTD_Const::C * dt * ((Ey(i + 1, 0) - Ey(i - 1, 0)) / (2.0 * dx) - (Ex(i, 1) - Ex(i, Nj - 1)) / (2.0 * dy));
        }
        Bx(Ni - 1, 0) -= FDTD_Const::C * dt * (Ez(Ni - 1, 1) - Ez(Ni - 1, Nj - 1)) / (2.0 * dy);
        Bz(Ni - 1, 0) -= FDTD_Const::C * dt * ((Ey(0, 0) - Ey(Ni - 2, 0)) / (2.0 * dx) - (Ex(Ni - 1, 1) - Ex(Ni - 1, Nj - 1)) / (2.0 * dy));

        // Basic parallelized cycles
        #pragma omp parallel private (i,j) default(shared)
        {
            #pragma omp for schedule(static)
            for (j = 1; j < Nj - 1; ++j)
            {
                for (i = 0; i < Ni; ++i)
                {
                    Bx(i, j) -= FDTD_Const::C * dt * (Ez(i, j + 1) - Ez(i, j - 1)) / (2.0 * dy);
                }
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj; ++j)
            {
                By(0, j) += FDTD_Const::C * dt * (Ez(1, j) - Ez(Ni - 1, j)) / (2.0 * dx);
                for (i = 1; i < Ni - 1; ++i)
                {
                    By(i, j) += FDTD_Const::C * dt * (Ez(i + 1, j) - Ez(i - 1, j)) / (2.0 * dx);
                }
                By(Ni - 1, j) += FDTD_Const::C * dt * (Ez(0, j) - Ez(Ni - 2, j)) / (2.0 * dx);
            }
            #pragma omp for schedule(static)
            for (j = 1; j < Nj - 1; ++j)
            {
                Bz(0, j) -= FDTD_Const::C * dt * ((Ey(1, j) - Ey(Ni - 1, j)) / (2.0 * dx) - (Ex(0, j + 1) - Ex(0, j - 1)) / (2.0 * dy));
                for (i = 1; i < Ni - 1; ++i)
                {
                    Bz(i, j) -= FDTD_Const::C * dt * ((Ey(i + 1, j) - Ey(i - 1, j)) / (2.0 * dx) - (Ex(i, j + 1) - Ex(i, j - 1)) / (2.0 * dy));
                }
                Bz(Ni - 1, j) -= FDTD_Const::C * dt * ((Ey(0, j) - Ey(Ni - 2, j)) / (2.0 * dx) - (Ex(Ni - 1, j + 1) - Ex(Ni - 1, j - 1)) / (2.0 * dy));
            }
        }
        // Right boundary conditions
        Bx(0, Nj - 1) -= FDTD_Const::C * dt * (Ez(0, 0) - Ez(0, Nj - 2)) / (2.0 * dy);
        Bz(0, Nj - 1) -= FDTD_Const::C * dt * ((Ey(1, Nj - 1) - Ey(Ni - 1, Nj - 1)) / (2.0 * dx) - (Ex(0, 0) - Ex(0, Nj - 2)) / (2.0 * dy));
        for (i = 1; i < Ni - 1; ++i)
        {
            Bx(i, Nj - 1) -= FDTD_Const::C * dt * (Ez(i, 0) - Ez(i, Nj - 2)) / (2.0 * dy);
            Bz(i, Nj - 1) -= FDTD_Const::C * dt * ((Ey(i + 1, Nj - 1) - Ey(i - 1, Nj - 1)) / (2.0 * dx) - (Ex(i, 0) - Ex(i, Nj - 2)) / (2.0 * dy));
        }
        Bx(Ni - 1, Nj - 1) -= FDTD_Const::C * dt * (Ez(Ni - 1, 0) - Ez(Ni - 1, Nj - 2)) / (2.0 * dy);
        Bz(Ni - 1, Nj - 1) -= FDTD_Const::C * dt * ((Ey(0, Nj - 1) - Ey(Ni - 2, Nj - 1)) / (2.0 * dx) - (Ex(Ni - 1, 0) - Ex(Ni - 1, Nj - 2)) / (2.0 * dy));
    }
}

void FDTD::shifted_update_field(const double& time)
{
    for (double t = 0; t <= time; t += dt)
    {
        int j;
        int i;

        // Basic parallelized cycles
        #pragma omp parallel private (i,j) default(shared)
        {
            #pragma omp for schedule(static)
            for (j = 0; j < Nj - 1; ++j)
            {
                for (i = 0; i < Ni; ++i)
                {
                    Bx(i, j) -= FDTD_Const::C * dt / 2.0 * (Ez(i, j + 1) - Ez(i, j)) / dy;
                }
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj; ++j)
            {
                for (i = 0; i < Ni - 1; ++i)
                {
                    By(i, j) += FDTD_Const::C * dt / 2.0 * (Ez(i + 1, j) - Ez(i, j)) / dx;
                }
                By(Ni - 1, j) += FDTD_Const::C * dt / 2.0 * (Ez(0, j) - Ez(Ni - 1, j)) / dx;
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj - 1; ++j)
            {
                for (i = 0; i < Ni - 1; ++i)
                {
                    Bz(i, j) -= FDTD_Const::C * dt / 2.0 * ((Ey(i + 1, j) - Ey(i, j)) / dx - (Ex(i, j + 1) - Ex(i, j)) / dy);
                }
                Bz(Ni - 1, j) -= FDTD_Const::C * dt / 2.0 * ((Ey(0, j) - Ey(Ni - 1, j)) / dx - (Ex(Ni - 1, j + 1) - Ex(Ni - 1, j)) / dy);
            }
        }
        // Right boundary conditions
        for (i = 0; i < Ni - 1; ++i)
        {
            Bx(i, Nj - 1) -= FDTD_Const::C * dt / 2.0 * (Ez(i, 0) - Ez(i, Nj - 1)) / dy;
            Bz(i, Nj - 1) -= FDTD_Const::C * dt / 2.0 * ((Ey(i + 1, Nj - 1) - Ey(i, Nj - 1)) / dx - (Ex(i, 0) - Ex(i, Nj - 1)) / dy);
        }
        Bx(Ni - 1, Nj - 1) -= FDTD_Const::C * dt / 2.0 * (Ez(Ni - 1, 0) - Ez(Ni - 1, Nj - 1)) / dy;
        Bz(Ni - 1, Nj - 1) -= FDTD_Const::C * dt / 2.0 * ((Ey(0, Nj - 1) - Ey(Ni - 1, Nj - 1)) / dx - (Ex(Ni - 1, 0) - Ex(Ni - 1, Nj - 1)) / dy);


        // Left boundary conditions
        Ex(0, 0) += FDTD_Const::C * dt * (Bz(0, 0) - Bz(0, Nj - 1)) / dy;
        Ez(0, 0) += FDTD_Const::C * dt * ((By(0, 0) - By(Ni - 1, 0)) / dx - (Bx(0, 0) - Bx(0, Nj - 1)) / dy);
        for (i = 1; i < Ni; ++i)
        {
            Ex(i, 0) += FDTD_Const::C * dt * (Bz(i, 0) - Bz(i, Nj - 1)) / dy;
            Ez(i, 0) += FDTD_Const::C * dt * ((By(i, 0) - By(i - 1, 0)) / dx - (Bx(i, 0) - Bx(i, Nj - 1)) / dy);
        }
        // Basic parallelized cycles
        #pragma omp parallel private (i,j) default(shared)
        {
            #pragma omp for schedule(static)
            for (j = 1; j < Nj; ++j)
            {
                for (i = 0; i < Ni; ++i)
                {
                    Ex(i, j) += FDTD_Const::C * dt * (Bz(i, j) - Bz(i, j - 1)) / dy;
                }
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj; ++j)
            {
                Ey(0, j) -= FDTD_Const::C * dt * (Bz(0, j) - Bz(Ni - 1, j)) / dx;
                for (i = 1; i < Ni; ++i)
                {
                    Ey(i, j) -= FDTD_Const::C * dt * (Bz(i, j) - Bz(i - 1, j)) / dx;
                }
            }
            #pragma omp for schedule(static)
            for (j = 1; j < Nj; ++j)
            {
                Ez(0, j) += FDTD_Const::C * dt * ((By(0, j) - By(Ni - 1, j)) / dx - (Bx(0, j) - Bx(0, j - 1)) / dy);
                for (i = 1; i < Ni; ++i)
                {
                    Ez(i, j) += FDTD_Const::C * dt * ((By(i, j) - By(i - 1, j)) / dx - (Bx(i, j) - Bx(i, j - 1)) / dy);
                }
            }
        }

        // Basic parallelized cycles
        #pragma omp parallel private (i,j) default(shared)
        {
            #pragma omp for schedule(static)
            for (j = 0; j < Nj - 1; ++j)
            {
                for (i = 0; i < Ni; ++i)
                {
                    Bx(i, j) -= FDTD_Const::C * dt / 2.0 * (Ez(i, j + 1) - Ez(i, j)) / dy;
                }
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj; ++j)
            {
                for (i = 0; i < Ni - 1; ++i)
                {
                    By(i, j) += FDTD_Const::C * dt / 2.0 * (Ez(i + 1, j) - Ez(i, j)) / dx;
                }
                By(Ni - 1, j) += FDTD_Const::C * dt / 2.0 * (Ez(0, j) - Ez(Ni - 1, j)) / dx;
            }
            #pragma omp for schedule(static)
            for (j = 0; j < Nj - 1; ++j)
            {
                for (i = 0; i < Ni - 1; ++i)
                {
                    Bz(i, j) -= FDTD_Const::C * dt / 2.0 * ((Ey(i + 1, j) - Ey(i, j)) / dx - (Ex(i, j + 1) - Ex(i, j)) / dy);
                }
                Bz(Ni - 1, j) -= FDTD_Const::C * dt / 2.0 * ((Ey(0, j) - Ey(Ni - 1, j)) / dx - (Ex(Ni - 1, j + 1) - Ex(Ni - 1, j)) / dy);
            }
        }
        // Right boundary conditions
        for (i = 0; i < Ni - 1; ++i)
        {
            Bx(i, Nj - 1) -= FDTD_Const::C * dt / 2.0 * (Ez(i, 0) - Ez(i, Nj - 1)) / dy;
            Bz(i, Nj - 1) -= FDTD_Const::C * dt / 2.0 * ((Ey(i + 1, Nj - 1) - Ey(i, Nj - 1)) / dx - (Ex(i, 0) - Ex(i, Nj - 1)) / dy);
        }
        Bx(Ni - 1, Nj - 1) -= FDTD_Const::C * dt / 2.0 * (Ez(Ni - 1, 0) - Ez(Ni - 1, Nj - 1)) / dy;
        Bz(Ni - 1, Nj - 1) -= FDTD_Const::C * dt / 2.0 * ((Ey(0, Nj - 1) - Ey(Ni - 1, Nj - 1)) / dx - (Ex(Ni - 1, 0) - Ex(Ni - 1, Nj - 1)) / dy);
    }
}
