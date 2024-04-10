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

void FDTD::set_sigma_x(int bounds_i[2], int bounds_j[2], int bounds_k[2],
    double SGm[2], std::function<int(int, int, int)> dist)
{
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                EsigmaX(i, j, k) = SGm[0] * pow((static_cast<double>(dist(i, j, k)))
                    / static_cast<double>(pml_size_i), FDTDconst::N);
                BsigmaX(i, j, k) = SGm[1] * pow((static_cast<double>(dist(i, j, k)) + 0.5)
                    / static_cast<double>(pml_size_i), FDTDconst::N);
            }
        }
    }
}
void FDTD::set_sigma_y(int bounds_i[2], int bounds_j[2], int bounds_k[2],
    double SGm[2], std::function<int(int, int, int)> dist)
{
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                EsigmaY(i, j, k) = SGm[0] * pow((static_cast<double>(dist(i, j, k)))
                    / static_cast<double>(pml_size_j), FDTDconst::N);
                BsigmaY(i, j, k) = SGm[1] * pow((static_cast<double>(dist(i, j, k)) + 0.5)
                    / static_cast<double>(pml_size_j), FDTDconst::N);
            }
        }
    }
}
void FDTD::set_sigma_z(int bounds_i[2], int bounds_j[2], int bounds_k[2],
    double SGm[2], std::function<int(int, int, int)> dist)
{
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                EsigmaZ(i, j, k) = SGm[0] * pow((static_cast<double>(dist(i, j, k)))
                    / static_cast<double>(pml_size_k), FDTDconst::N);
                BsigmaZ(i, j, k) = SGm[1] * pow((static_cast<double>(dist(i, j, k)) + 0.5)
                    / static_cast<double>(pml_size_k), FDTDconst::N);
            }
        }
    }
}



FDTD::FDTD(Parameters _parameters, double _dt, double _pml_percent) : 
    parameters(_parameters), dt(_dt), pml_percent(_pml_percent)
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
    Exy = Exz = Eyx = Eyz = Ezx = Ezy = Field(parameters.Ni, parameters.Nj, parameters.Nk);
    Bxy = Bxz = Byx = Byz = Bzx = Bzy = Field(parameters.Ni, parameters.Nj, parameters.Nk);
    EsigmaX = EsigmaY = EsigmaZ = BsigmaX = BsigmaY = BsigmaZ = Field(parameters.Ni, parameters.Nj, parameters.Nk);

    pml_size_i = static_cast<int>(static_cast<double>(parameters.Ni) * pml_percent);
    pml_size_j = static_cast<int>(static_cast<double>(parameters.Nj) * pml_percent);
    pml_size_k = static_cast<int>(static_cast<double>(parameters.Nk) * pml_percent);
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

void FDTD::update_E(int bounds_i[2], int bounds_j[2], int bounds_k[2], int t)
{
    double dx = parameters.dx;
    double dy = parameters.dy;
    double dz = parameters.dz;

#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                Ex(i, j, k) = Ex(i, j, k) - 4.0 * M_PI * dt * Jx[t](i, j, k) + FDTDconst::C * dt * ((Bz(i, j, k) - Bz(i, j - 1, k)) / dy - (By(i, j, k) - By(i, j, k - 1)) / dz);
                Ey(i, j, k) = Ey(i, j, k) - 4.0 * M_PI * dt * Jy[t](i, j, k) + FDTDconst::C * dt * ((Bx(i, j, k) - Bx(i, j, k - 1)) / dz - (Bz(i, j, k) - Bz(i - 1, j, k)) / dx);
                Ez(i, j, k) = Ez(i, j, k) - 4.0 * M_PI * dt * Jz[t](i, j, k) + FDTDconst::C * dt * ((By(i, j, k) - By(i - 1, j, k)) / dx - (Bx(i, j, k) - Bx(i, j - 1, k)) / dy);
            }
        }
    }
}

void FDTD::update_B(int bounds_i[2], int bounds_j[2], int bounds_k[2], int t)
{
    double dx = parameters.dx;
    double dy = parameters.dy;
    double dz = parameters.dz;

#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                Bx(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ey(i, j, k + 1) - Ey(i, j, k)) / dz - (Ez(i, j + 1, k) - Ez(i, j, k)) / dy);
                By(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ez(i + 1, j, k) - Ez(i, j, k)) / dx - (Ex(i, j, k + 1) - Ex(i, j, k)) / dz);
                Bz(i, j, k) += FDTDconst::C * dt / 2.0 * ((Ex(i, j + 1, k) - Ex(i, j, k)) / dy - (Ey(i + 1, j, k) - Ey(i, j, k)) / dx);
            }
        }
    }
}

double FDTD::PMLcoef(double sigma, double constant)
{
    return exp(-sigma * dt * FDTDconst::C);// / constant);
}

void FDTD::update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2])
{
    double dx = parameters.dx;
    double dy = parameters.dy;
    double dz = parameters.dz;
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    { 
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                if (EsigmaX(i, j, k) != 0)
                {
                    Eyx(i, j, k) = Eyx(i, j, k) * PMLcoef(EsigmaX(i, j, k), FDTDconst::EPS0) -
                        (1.0 - PMLcoef(EsigmaX(i, j, k), FDTDconst::EPS0)) / (EsigmaX(i, j, k) * dx) * //FDTDconst::C *
                        (Bz(i, j, k) - Bz(i - 1, j, k));
                    Ezx(i, j, k) = Ezx(i, j, k) * PMLcoef(EsigmaX(i, j, k), FDTDconst::EPS0) +
                        (1.0 - PMLcoef(EsigmaX(i, j, k), FDTDconst::EPS0)) / (EsigmaX(i, j, k) * dx) * //FDTDconst::C *
                        (By(i, j, k) - By(i - 1, j, k));
                }
                if (EsigmaY(i, j, k) != 0)
                {
                    Exy(i, j, k) = Exy(i, j, k) * PMLcoef(EsigmaY(i, j, k), FDTDconst::EPS0) +
                        (1.0 - PMLcoef(EsigmaY(i, j, k), FDTDconst::EPS0)) / (EsigmaY(i, j, k) * dy) * //FDTDconst::C *
                        (Bz(i, j, k) - Bz(i, j - 1, k));
                    Ezy(i, j, k) = Ezy(i, j, k) * PMLcoef(EsigmaY(i, j, k), FDTDconst::EPS0) -
                        (1.0 - PMLcoef(EsigmaY(i, j, k), FDTDconst::EPS0)) / (EsigmaY(i, j, k) * dy) * //FDTDconst::C *
                        (Bx(i, j, k) - Bx(i, j - 1, k));
                }
                if (EsigmaZ(i, j, k) != 0)
                {
                    Exz(i, j, k) = Exz(i, j, k) * PMLcoef(EsigmaZ(i, j, k), FDTDconst::EPS0) -
                        (1.0 - PMLcoef(EsigmaZ(i, j, k), FDTDconst::EPS0)) / (EsigmaZ(i, j, k) * dz) * //FDTDconst::C *
                        (By(i, j, k) - By(i, j, k - 1));
                    Eyz(i, j, k) = Eyz(i, j, k) * PMLcoef(EsigmaZ(i, j, k), FDTDconst::EPS0) +
                        (1.0 - PMLcoef(EsigmaZ(i, j, k), FDTDconst::EPS0)) / (EsigmaZ(i, j, k) * dz) * //FDTDconst::C *
                        (Bx(i, j, k) - Bx(i, j, k - 1));
                }
                Ex(i, j, k) = Exz(i, j, k) + Exy(i, j, k);
                Ey(i, j, k) = Eyx(i, j, k) + Eyz(i, j, k);
                Ez(i, j, k) = Ezy(i, j, k) + Ezx(i, j, k);
            }
        }
    }
}

void FDTD::update_B_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2])
{
    double dx = parameters.dx;
    double dy = parameters.dy;
    double dz = parameters.dz;
    
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                if (BsigmaX(i, j, k) != 0.0)
                {
                    Byx(i, j, k) = Byx(i, j, k) * PMLcoef(BsigmaX(i, j, k), FDTDconst::MU0) +
                        (1.0 - PMLcoef(BsigmaX(i, j, k), FDTDconst::MU0)) / (BsigmaX(i, j, k) * dx) * //FDTDconst::C *
                        (Ez(i + 1, j, k) - Ez(i, j, k));
                    Bzx(i, j, k) = Bzx(i, j, k) * PMLcoef(BsigmaX(i, j, k), FDTDconst::MU0) -
                        (1.0 - PMLcoef(BsigmaX(i, j, k), FDTDconst::MU0)) / (BsigmaX(i, j, k) * dx) * //FDTDconst::C *
                        (Ey(i + 1, j, k) - Ey(i, j, k));
                }
                if (BsigmaY(i, j, k) != 0.0)
                {
                    Bxy(i, j, k) = Bxy(i, j, k) * PMLcoef(BsigmaY(i, j, k), FDTDconst::MU0) -
                        (1.0 - PMLcoef(BsigmaY(i, j, k), FDTDconst::MU0)) / (BsigmaY(i, j, k) * dy) * //FDTDconst::C *
                        (Ez(i, j + 1, k) - Ez(i, j, k));
                    Bzy(i, j, k) = Bzy(i, j, k) * PMLcoef(BsigmaY(i, j, k), FDTDconst::MU0) +
                        (1.0 - PMLcoef(BsigmaY(i, j, k), FDTDconst::MU0)) / (BsigmaY(i, j, k) * dy) * //FDTDconst::C *
                        (Ex(i, j + 1, k) - Ex(i, j, k));
                }
                if (BsigmaZ(i, j, k) != 0.0)
                {
                    Bxz(i, j, k) = Bxz(i, j, k) * PMLcoef(BsigmaZ(i, j, k), FDTDconst::MU0) +
                        (1.0 - PMLcoef(BsigmaZ(i, j, k), FDTDconst::MU0)) / (BsigmaZ(i, j, k) * dz) * //FDTDconst::C *
                        (Ey(i, j, k + 1) - Ey(i, j, k));
                    Byz(i, j, k) = Byz(i, j, k) * PMLcoef(BsigmaZ(i, j, k), FDTDconst::MU0) -
                        (1.0 - PMLcoef(BsigmaZ(i, j, k), FDTDconst::MU0)) / (BsigmaZ(i, j, k) * dz) * //FDTDconst::C *
                        (Ex(i, j, k + 1) - Ex(i, j, k));
                }
                Bx(i, j, k) = Bxy(i, j, k) + Bxz(i, j, k);
                By(i, j, k) = Byz(i, j, k) + Byx(i, j, k);
                Bz(i, j, k) = Bzx(i, j, k) + Bzy(i, j, k);
            }
        }
    }
}

std::vector<std::vector<Field>> FDTD::update_fields(const int time)
{
    if (time < 0)
    {
        throw std::exception("ERROR: Invalid update field argument");
    }
    std::vector<std::vector<Field>> return_data;

    if (pml_percent == 0.0)
    {
        int size_i_main[2] = { 0, parameters.Ni };
        int size_j_main[2] = { 0, parameters.Nj };
        int size_k_main[2] = { 0, parameters.Nk };
        for (int t = 0; t < time; t++)
        {
            std::cout << "Iteration: " << t + 1 << std::endl;

            update_B(size_i_main, size_j_main, size_k_main, t);
            update_E(size_i_main, size_j_main, size_k_main, t);
            update_B(size_i_main, size_j_main, size_k_main, t);

            std::vector<Field> new_iteration{ Ex, Ey, Ez, Bx, By, Bz };
            return_data.push_back(new_iteration);
        }
        return return_data;
    }

    int size_i_main[2] = { pml_size_i, parameters.Ni - pml_size_i };
    int size_j_main[2] = { pml_size_j, parameters.Nj - pml_size_j };
    int size_k_main[2] = { pml_size_k, parameters.Nk - pml_size_k };

    /*int size_i_main[2] = { 0, parameters.Ni };
    int size_j_main[2] = { 0, parameters.Nj };
    int size_k_main[2] = { 0, parameters.Nk };*/

    int size_i_solid[2] = { 0, parameters.Ni };
    int size_j_solid[2] = { 0, parameters.Nj };
    int size_k_solid[2] = { 0, parameters.Nk };

    int size_xy_lower_k_pml[2] = { 0, pml_size_k };
    int size_xy_upper_k_pml[2] = { parameters.Nk - pml_size_k, parameters.Nk };

    int size_yz_lower_i_pml[2] = { 0, pml_size_i };
    int size_yz_upper_i_pml[2] = { parameters.Ni - pml_size_i, parameters.Ni };

    int size_zx_lower_j_pml[2] = { 0, pml_size_j };
    int size_zx_upper_j_pml[2] = { parameters.Nj - pml_size_j, parameters.Nj };

    std::function<int(int, int, int)> calc_distant_i_up = [=](int i, int j, int k) { 
        return i + 1 + pml_size_i - parameters.Ni;
    };
    std::function<int(int, int, int)> calc_distant_j_up = [=](int i, int j, int k) { 
        return j + 1 + pml_size_j - parameters.Nj;
    };
    std::function<int(int, int, int)> calc_distant_k_up = [=](int i, int j, int k) { 
        return k + 1 + pml_size_k - parameters.Nk;
    };

    std::function<int(int, int, int)> calc_distant_i_low = [=](int i, int j, int k) { 
        return pml_size_i - i;
    };
    std::function<int(int, int, int)> calc_distant_j_low = [=](int i, int j, int k) { 
        return pml_size_j - j;
    };
    std::function<int(int, int, int)> calc_distant_k_low = [=](int i, int j, int k) { 
        return pml_size_k - k;
    };

    double SGm_Ex = -(FDTDconst::N + 1.0) / 2.0 * log(FDTDconst::R) * FDTDconst::C * FDTDconst::EPS0 
        / (static_cast<double>(pml_size_i) * parameters.dx);
    double SGm_Ey = -(FDTDconst::N + 1.0) / 2.0 * log(FDTDconst::R) * FDTDconst::C * FDTDconst::EPS0 
        / (static_cast<double>(pml_size_j) * parameters.dy);
    double SGm_Ez = -(FDTDconst::N + 1.0) / 2.0 * log(FDTDconst::R) * FDTDconst::C * FDTDconst::EPS0 
        / (static_cast<double>(pml_size_k) * parameters.dz);

    double SGm_Bx = -(FDTDconst::N + 1.0) / 2.0 * log(FDTDconst::R) * FDTDconst::C * FDTDconst::MU0 
        / (static_cast<double>(pml_size_i) * parameters.dx);
    double SGm_By = -(FDTDconst::N + 1.0) / 2.0 * log(FDTDconst::R) * FDTDconst::C * FDTDconst::MU0 
        / (static_cast<double>(pml_size_j) * parameters.dy);
    double SGm_Bz = -(FDTDconst::N + 1.0) / 2.0 * log(FDTDconst::R) * FDTDconst::C * FDTDconst::MU0 
        / (static_cast<double>(pml_size_k) * parameters.dz);

    double SG_x[] = { SGm_Ex, SGm_Bx };
    double SG_y[] = { SGm_Ey, SGm_By };
    double SG_z[] = { SGm_Ez, SGm_Bz };

    set_sigma_z(size_i_solid, size_j_solid, size_xy_lower_k_pml, SG_z, calc_distant_k_low);
    set_sigma_y(size_i_solid, size_zx_lower_j_pml, size_k_solid, SG_y, calc_distant_j_low);
    set_sigma_x(size_yz_lower_i_pml, size_j_solid, size_k_solid, SG_x, calc_distant_i_low);

    set_sigma_z(size_i_solid, size_j_solid, size_xy_upper_k_pml, SG_z, calc_distant_k_up);
    set_sigma_y(size_i_solid, size_zx_upper_j_pml, size_k_solid, SG_y, calc_distant_j_up);
    set_sigma_x(size_yz_upper_i_pml, size_j_solid, size_k_solid, SG_x, calc_distant_i_up);


    double max_val_1 = 0.0;
    double max_val_2 = 0.0;

    int tmax1 = 0;
    int tmax2 = 0;

    for (int t = 0; t < time; t++)
    {
        std::cout << "Iteration: " << t + 1 << std::endl;

        update_B(size_i_main, size_j_main, size_k_main, t);

        update_B_PML(size_i_solid, size_j_solid, size_xy_lower_k_pml);
        update_B_PML(size_i_solid, size_zx_lower_j_pml, size_k_solid);
        update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_solid);

        update_B_PML(size_i_solid, size_j_solid, size_xy_upper_k_pml);
        update_B_PML(size_i_solid, size_zx_upper_j_pml, size_k_solid);
        update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_solid);

        update_E(size_i_main, size_j_main, size_k_main, t);

        update_E_PML(size_i_solid, size_j_solid, size_xy_lower_k_pml);
        update_E_PML(size_i_solid, size_zx_lower_j_pml, size_k_solid);
        update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_solid);

        update_E_PML(size_i_solid, size_j_solid, size_xy_upper_k_pml);
        update_E_PML(size_i_solid, size_zx_upper_j_pml, size_k_solid);
        update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_solid);

        update_B(size_i_main, size_j_main, size_k_main, t);

        if (t < 110 && t > 35)
        {
            for (int i = 0; i < parameters.Ni - pml_size_i; i++)
            {
                for (int j = 0; j < parameters.Nj - pml_size_j; j++)
                {
                    for (int k = 0; k < parameters.Nk - pml_size_k; k++)
                    {
                        if (fabs(Ex(i, j, k)) > fabs(max_val_1))
                        {
                            max_val_1 = Ex(i, j, k);
                            tmax1 = t;
                        }
                        if (fabs(Ey(i, j, k)) > fabs(max_val_1))
                        {
                            max_val_1 = Ey(i, j, k);
                            tmax1 = t;
                        }
                        if (fabs(Ez(i, j, k)) > fabs(max_val_1))
                        {
                            max_val_1 = Ez(i, j, k);
                            tmax1 = t;
                        }
                        if (fabs(Bx(i, j, k)) > fabs(max_val_1))
                        {
                            max_val_1 = Bx(i, j, k);
                            tmax1 = t;
                        }
                        if (fabs(By(i, j, k)) > fabs(max_val_1))
                        {
                            max_val_1 = By(i, j, k);
                            tmax1 = t;
                        }
                        if (fabs(Bz(i, j, k)) > fabs(max_val_1))
                        {
                            max_val_1 = Bz(i, j, k);
                            tmax1 = t;
                        }
                    }
                }
            }
        }
        if (t > 200)
        {
            for (int i = 0; i < parameters.Ni - pml_size_i; i++)
            {
                for (int j = 0; j < parameters.Nj - pml_size_j; j++)
                {
                    for (int k = 0; k < parameters.Nk - pml_size_k; k++)
                    {
                        if (fabs(Ex(i, j, k)) > fabs(max_val_2))
                        {
                            max_val_2 = Ex(i, j, k);
                            tmax2 = t;
                        }
                        if (fabs(Ey(i, j, k)) > fabs(max_val_2))
                        {
                            max_val_2 = Ey(i, j, k);
                            tmax2 = t;
                        }
                        if (fabs(Ez(i, j, k)) > fabs(max_val_2))
                        {
                            max_val_2 = Ez(i, j, k);
                            tmax2 = t;
                        }
                        if (fabs(Bx(i, j, k)) > fabs(max_val_2))
                        {
                            max_val_2 = Bx(i, j, k);
                            tmax2 = t;
                        }
                        if (fabs(By(i, j, k)) > fabs(max_val_2))
                        {
                            max_val_2 = By(i, j, k);
                            tmax2 = t;
                        }
                        if (fabs(Bz(i, j, k)) > fabs(max_val_2))
                        {
                            max_val_2 = Bz(i, j, k);
                            tmax2 = t;
                        }
                        
                    }
                }
            }
        }

        std::vector<Field> new_iteration{ Ex, Ey, Ez, Bx, By, Bz };
        return_data.push_back(new_iteration);
        //if (t == 0) break;
    }

    std::cout << "before: " << max_val_1 << " | " << tmax1 << std::endl;
    std::cout << "after: " << max_val_2 << " | " << tmax2 << std::endl;

    return return_data;
}
