#include "FDTD.h"

#include <iostream>
#include <exception>
#include <cmath>

//#define __DEBUG_PML__
//#define __OUTPUT_SIGMA__

Field::Field(const int _Ni = 1, const int _Nj = 1, const int _Nk = 1) 
    : Ni(_Ni), Nj(_Nj), Nk(_Nk)
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

    int truly_i = (Ni - 1) * i_isMinusOne + i *
        !(i_isMinusOne || i_isNi);
    int truly_j = (Nj - 1) * j_isMinusOne + j *
        !(j_isMinusOne || j_isNj);
    int truly_k = (Nk - 1) * k_isMinusOne + k *
        !(k_isMinusOne || k_isNk);

    int index = truly_i + truly_j * Ni + truly_k * Ni * Nj;
    return field[index];
}

void FDTD::set_sigma_x(int bounds_i[2], int bounds_j[2], int bounds_k[2],
    double SGm, std::function<int(int, int, int)> dist)
{
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                EsigmaX(i, j, k) = SGm *
                    std::pow((static_cast<double>(dist(i, j, k))) /
                    static_cast<double>(pml_size_i), FDTDconst::N);
                BsigmaX(i, j, k) = SGm *
                    std::pow((static_cast<double>(dist(i, j, k))) /
                    static_cast<double>(pml_size_i), FDTDconst::N);
            }
        }
    }
}
void FDTD::set_sigma_y(int bounds_i[2], int bounds_j[2], int bounds_k[2],
    double SGm, std::function<int(int, int, int)> dist)
{
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                EsigmaY(i, j, k) = SGm *
                    std::pow((static_cast<double>(dist(i, j, k))) /
                        static_cast<double>(pml_size_j), FDTDconst::N);
                BsigmaY(i, j, k) = SGm *
                    std::pow((static_cast<double>(dist(i, j, k))) /
                        static_cast<double>(pml_size_j), FDTDconst::N);
            }
        }
    }
}
void FDTD::set_sigma_z(int bounds_i[2], int bounds_j[2], int bounds_k[2],
    double SGm, std::function<int(int, int, int)> dist)
{
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                EsigmaZ(i, j, k) = SGm *
                    std::pow((static_cast<double>(dist(i, j, k))) /
                        static_cast<double>(pml_size_k), FDTDconst::N);
                BsigmaZ(i, j, k) = SGm *
                    std::pow((static_cast<double>(dist(i, j, k))) /
                        static_cast<double>(pml_size_k), FDTDconst::N);
            }
        }
    }
}

FDTD::FDTD(Parameters _parameters, double _dt, double _pml_percent) : 
    parameters(_parameters), dt(_dt), pml_percent(_pml_percent)
{
    if (parameters.Ni <= 0 ||
        parameters.Nj <= 0 ||
        parameters.Nk <= 0 ||
        dt <= 0)
    {
        throw std::invalid_argument("ERROR: invalid parameters");
    }
    Ex = Ey = Ez = Bx = By = Bz 
        = Field(parameters.Ni, parameters.Nj, parameters.Nk);
    Exy = Exz = Eyx = Eyz = Ezx = Ezy 
        = Field(parameters.Ni, parameters.Nj, parameters.Nk);
    Bxy = Bxz = Byx = Byz = Bzx = Bzy 
        = Field(parameters.Ni, parameters.Nj, parameters.Nk);
    EsigmaX = EsigmaY = EsigmaZ = BsigmaX = BsigmaY = BsigmaZ 
        = Field(parameters.Ni, parameters.Nj, parameters.Nk);

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

    default: throw std::logic_error("ERROR: Invalid field component");
    }
}

std::vector<Field>& FDTD::get_current(Component this_current)
{
    switch (this_current)
    {
    case Component::JX: return Jx;

    case Component::JY: return Jy;

    case Component::JZ: return Jz;

    default: throw std::logic_error("ERROR: Invalid current component");
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
                Ex(i, j, k) = Ex(i, j, k) - 4.0 * FDTDconst::PI * dt * Jx[t](i, j, k) + 
                    FDTDconst::C * dt * ((Bz(i, j, k) - Bz(i, j - 1, k)) / dy - 
                                         (By(i, j, k) - By(i, j, k - 1)) / dz);
                Ey(i, j, k) = Ey(i, j, k) - 4.0 * FDTDconst::PI * dt * Jy[t](i, j, k) + 
                    FDTDconst::C * dt * ((Bx(i, j, k) - Bx(i, j, k - 1)) / dz - 
                                         (Bz(i, j, k) - Bz(i - 1, j, k)) / dx);
                Ez(i, j, k) = Ez(i, j, k) - 4.0 * FDTDconst::PI * dt * Jz[t](i, j, k) + 
                    FDTDconst::C * dt * ((By(i, j, k) - By(i - 1, j, k)) / dx - 
                                         (Bx(i, j, k) - Bx(i, j - 1, k)) / dy);
            }
        }
    }
}

void FDTD::update_B(int bounds_i[2], int bounds_j[2], int bounds_k[2])
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
                Bx(i, j, k) += FDTDconst::C * dt / 2.0 * 
                    ((Ey(i, j, k + 1) - Ey(i, j, k)) / dz - 
                     (Ez(i, j + 1, k) - Ez(i, j, k)) / dy);
                By(i, j, k) += FDTDconst::C * dt / 2.0 * 
                    ((Ez(i + 1, j, k) - Ez(i, j, k)) / dx - 
                     (Ex(i, j, k + 1) - Ex(i, j, k)) / dz);
                Bz(i, j, k) += FDTDconst::C * dt / 2.0 * 
                    ((Ex(i, j + 1, k) - Ex(i, j, k)) / dy - 
                     (Ey(i + 1, j, k) - Ey(i, j, k)) / dx);
            }
        }
    }
}

double FDTD::PMLcoef(double sigma)
{
    return std::exp(-sigma * dt * FDTDconst::C);
}

void FDTD::update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2])
{
    double dx = parameters.dx;
    double dy = parameters.dy;
    double dz = parameters.dz;

    double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    { 
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                if (EsigmaX(i, j, k) != 0.0)
                    PMLcoef2_x = (1.0 - PMLcoef(EsigmaX(i, j, k))) / (EsigmaX(i, j, k) * dx);
                else
                    PMLcoef2_x = FDTDconst::C * dt / dx;

                if (EsigmaY(i, j, k) != 0.0)
                    PMLcoef2_y = (1.0 - PMLcoef(EsigmaY(i, j, k))) / (EsigmaY(i, j, k) * dy);
                else
                    PMLcoef2_y = FDTDconst::C * dt / dy;

                if (EsigmaZ(i, j, k) != 0.0)
                    PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ(i, j, k))) / (EsigmaZ(i, j, k) * dz);
                else
                    PMLcoef2_z = FDTDconst::C * dt / dz;

                Eyx(i, j, k) = Eyx(i, j, k) * PMLcoef(EsigmaX(i, j, k)) -
                    PMLcoef2_x * (Bz(i, j, k) - Bz(i - 1, j, k));
                Ezx(i, j, k) = Ezx(i, j, k) * PMLcoef(EsigmaX(i, j, k)) +
                    PMLcoef2_x * (By(i, j, k) - By(i - 1, j, k));

                Exy(i, j, k) = Exy(i, j, k) * PMLcoef(EsigmaY(i, j, k)) +
                    PMLcoef2_y * (Bz(i, j, k) - Bz(i, j - 1, k));
                Ezy(i, j, k) = Ezy(i, j, k) * PMLcoef(EsigmaY(i, j, k)) -
                    PMLcoef2_y * (Bx(i, j, k) - Bx(i, j - 1, k));

                Exz(i, j, k) = Exz(i, j, k) * PMLcoef(EsigmaZ(i, j, k)) -
                    PMLcoef2_z * (By(i, j, k) - By(i, j, k - 1));
                Eyz(i, j, k) = Eyz(i, j, k) * PMLcoef(EsigmaZ(i, j, k)) +
                    PMLcoef2_z * (Bx(i, j, k) - Bx(i, j, k - 1));

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

    double PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;
    
#pragma omp parallel for collapse(2)
    for (int i = bounds_i[0]; i < bounds_i[1]; i++)
    {
        for (int j = bounds_j[0]; j < bounds_j[1]; j++)
        {
            for (int k = bounds_k[0]; k < bounds_k[1]; k++)
            {
                if (BsigmaX(i, j, k) != 0.0)
                    PMLcoef2_x = (1.0 - PMLcoef(BsigmaX(i, j, k))) / (BsigmaX(i, j, k) * dx);
                else
                    PMLcoef2_x = FDTDconst::C * dt / dx;

                if (BsigmaY(i, j, k) != 0.0)
                    PMLcoef2_y = (1.0 - PMLcoef(BsigmaY(i, j, k))) / (BsigmaY(i, j, k) * dy);
                else
                    PMLcoef2_y = FDTDconst::C * dt / dy;

                if (BsigmaZ(i, j, k) != 0.0)
                    PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ(i, j, k))) / (BsigmaZ(i, j, k) * dz);
                else
                    PMLcoef2_z = FDTDconst::C * dt / dz;

                Byx(i, j, k) = Byx(i, j, k) * PMLcoef(BsigmaX(i, j, k)) +
                    PMLcoef2_x * (Ez(i + 1, j, k) - Ez(i, j, k));
                Bzx(i, j, k) = Bzx(i, j, k) * PMLcoef(BsigmaX(i, j, k)) -
                    PMLcoef2_x * (Ey(i + 1, j, k) - Ey(i, j, k));

                Bxy(i, j, k) = Bxy(i, j, k) * PMLcoef(BsigmaY(i, j, k)) -
                    PMLcoef2_y * (Ez(i, j + 1, k) - Ez(i, j, k));
                Bzy(i, j, k) = Bzy(i, j, k) * PMLcoef(BsigmaY(i, j, k)) +
                    PMLcoef2_y * (Ex(i, j + 1, k) - Ex(i, j, k));

                Bxz(i, j, k) = Bxz(i, j, k) * PMLcoef(BsigmaZ(i, j, k)) +
                    PMLcoef2_z * (Ey(i, j, k + 1) - Ey(i, j, k));
                Byz(i, j, k) = Byz(i, j, k) * PMLcoef(BsigmaZ(i, j, k)) -
                    PMLcoef2_z * (Ex(i, j, k + 1) - Ex(i, j, k));

                Bx(i, j, k) = Bxy(i, j, k) + Bxz(i, j, k);
                By(i, j, k) = Byz(i, j, k) + Byx(i, j, k);
                Bz(i, j, k) = Bzx(i, j, k) + Bzy(i, j, k);
            }
        }
    }
}

double FDTD::calc_reflection(Field E_start[3], Field B_start[3],
    Field E_final[3], Field B_final[3])
{
    double energy_start = 0.0;
    double energy_final = 0.0;

#pragma omp parallel for collapse(2)
    for (int i = pml_size_i; i < parameters.Ni - pml_size_i; i++)
    {
        for (int j = pml_size_j; j < parameters.Nj - pml_size_j; j++)
        {
            for (int k = pml_size_k; k < parameters.Nk - pml_size_k; k++)
            {
                energy_start += std::pow(E_start[0](i, j, k), 2.0) +
                    std::pow(E_start[1](i, j, k), 2.0) +
                    std::pow(E_start[2](i, j, k), 2.0) +
                    std::pow(B_start[0](i, j, k), 2.0) +
                    std::pow(B_start[1](i, j, k), 2.0) +
                    std::pow(B_start[2](i, j, k), 2.0);

                energy_final += std::pow(E_final[0](i, j, k), 2.0) +
                    std::pow(E_final[1](i, j, k), 2.0) +
                    std::pow(E_final[2](i, j, k), 2.0) +
                    std::pow(B_final[0](i, j, k), 2.0) +
                    std::pow(B_final[1](i, j, k), 2.0) +
                    std::pow(B_final[2](i, j, k), 2.0);
            }
        }
    }

    return std::sqrt(energy_final / energy_start);
}

double FDTD::calc_max_value(Field& field)
{
    double max_val = 0.0;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < parameters.Ni - pml_size_i; i++)
    {
        for (int j = 0; j < parameters.Nj - pml_size_j; j++)
        {
            for (int k = 0; k < parameters.Nk - pml_size_k; k++)
            {
                if (std::abs(field(i, j, k)) > std::abs(max_val))
                {
                    max_val = field(i, j, k);
                }
            }
        }
    }
    return max_val;
}

std::vector<std::vector<Field>> FDTD::update_fields(const int time)
{
    if (time < 0)
    {
        throw std::invalid_argument("ERROR: Invalid update field argument");
    }
    std::vector<std::vector<Field>> return_data;

    if (pml_percent == 0.0)
    {
        int size_i_main[2] = { 0, parameters.Ni };
        int size_j_main[2] = { 0, parameters.Nj };
        int size_k_main[2] = { 0, parameters.Nk };
        for (int t = 0; t < time; t++)
        {
            update_B(size_i_main, size_j_main, size_k_main);
            update_E(size_i_main, size_j_main, size_k_main, t);
            update_B(size_i_main, size_j_main, size_k_main);

            std::vector<Field> new_iteration{ Ex, Ey, Ez, Bx, By, Bz };
            return_data.push_back(new_iteration);
        }
        return return_data;
    }
    
    // Defining areas of computation
    int size_i_main[] = { pml_size_i, parameters.Ni - pml_size_i };
    int size_j_main[] = { pml_size_j, parameters.Nj - pml_size_j };
    int size_k_main[] = { pml_size_k, parameters.Nk - pml_size_k };

    int size_i_solid[] = { 0, parameters.Ni };
    int size_j_solid[] = { 0, parameters.Nj };
    int size_k_solid[] = { 0, parameters.Nk };

    int size_i_part_from_start[] = { 0, parameters.Ni - pml_size_i };
    int size_i_part_from_end[] = { pml_size_i, parameters.Ni };

    int size_k_part_from_start[] = { 0, parameters.Nk - pml_size_k };
    int size_k_part_from_end[] = { pml_size_k, parameters.Nk };

    int size_xy_lower_k_pml[] = { 0, pml_size_k };
    int size_xy_upper_k_pml[] = { parameters.Nk - pml_size_k, parameters.Nk };

    int size_yz_lower_i_pml[] = { 0, pml_size_i };
    int size_yz_upper_i_pml[] = { parameters.Ni - pml_size_i, parameters.Ni };

    int size_zx_lower_j_pml[] = { 0, pml_size_j };
    int size_zx_upper_j_pml[] = { parameters.Nj - pml_size_j, parameters.Nj };

    // Definition of functions for calculating the distance to the interface
    std::function<int(int, int, int)> calc_distant_i_up = 
        [=](int i, int j, int k) { 
        return i + 1 + pml_size_i - parameters.Ni;
    };
    std::function<int(int, int, int)> calc_distant_j_up = 
        [=](int i, int j, int k) { 
        return j + 1 + pml_size_j - parameters.Nj;
    };
    std::function<int(int, int, int)> calc_distant_k_up = 
        [=](int i, int j, int k) { 
        return k + 1 + pml_size_k - parameters.Nk;
    };

    std::function<int(int, int, int)> calc_distant_i_low = 
        [=](int i, int j, int k) { 
        return pml_size_i - i;
    };
    std::function<int(int, int, int)> calc_distant_j_low = 
        [=](int i, int j, int k) { 
        return pml_size_j - j;
    };
    std::function<int(int, int, int)> calc_distant_k_low = 
        [=](int i, int j, int k) { 
        return pml_size_k - k;
    };

    // Calculation of maximum permittivity
    double SGm_x = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
        / (static_cast<double>(pml_size_i) * parameters.dx);
    double SGm_y = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
        / (static_cast<double>(pml_size_j) * parameters.dy);
    double SGm_z = -(FDTDconst::N + 1.0) / 2.0 * std::log(FDTDconst::R)
        / (static_cast<double>(pml_size_k) * parameters.dz);

    // Calculation of permittivity in the cells
    set_sigma_z(size_i_solid, size_j_solid, size_xy_lower_k_pml, 
                SGm_z, calc_distant_k_low);
    set_sigma_y(size_i_solid, size_zx_lower_j_pml, size_k_solid, 
                SGm_y, calc_distant_j_low);
    set_sigma_x(size_yz_lower_i_pml, size_j_solid, size_k_solid, 
                SGm_x, calc_distant_i_low);

    set_sigma_z(size_i_solid, size_j_solid, size_xy_upper_k_pml, 
                SGm_z, calc_distant_k_up);
    set_sigma_y(size_i_solid, size_zx_upper_j_pml, size_k_solid, 
                SGm_y, calc_distant_j_up);
    set_sigma_x(size_yz_upper_i_pml, size_j_solid, size_k_solid, 
                SGm_x, calc_distant_i_up);

#ifdef __DEBUG_PML__
    double max_val_1 = 0.0;
    double max_val_2 = 0.0;
    int t_start = 50;
    int t_final = 400;
#endif // __DEBUG_PML__

    for (int t = 0; t < time; t++)
    {
        std::cout << "Iteration: " << t + 1 << std::endl;

        update_B(size_i_main, size_j_main, size_k_main);

        update_B_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
        update_B_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
        update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

        update_B_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
        update_B_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
        update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

        update_E(size_i_main, size_j_main, size_k_main, t);

        update_E_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
        update_E_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
        update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

        update_E_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
        update_E_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
        update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

        update_B(size_i_main, size_j_main, size_k_main);

#ifdef __DEBUG_PML__
        if (t == t_start)
        {
            max_val_1 = calc_max_value(Ex);
        }
        if (t == t_final)
        {
            max_val_2 = calc_max_value(Ex);
        }

#endif //__DEBUG_PML__

#ifndef __OUTPUT_SIGMA__
        std::vector<Field> new_iteration{ Ex, Ey, Ez, Bx, By, Bz };
#endif // __OUTPUT_SIGMA__

#ifdef __OUTPUT_SIGMA__

        EsigmaX(parameters.Ni / 2, 1, 1) = 100;

        std::vector<Field> new_iteration{ EsigmaX, EsigmaY, EsigmaZ,
                                          BsigmaX, BsigmaY, BsigmaZ };
        return_data.push_back(new_iteration);
        if (t == 0) break;
#endif // __OUTPUT_SIGMA__

        return_data.push_back(new_iteration);
    }

#ifdef __DEBUG_PML__
    Field E_start[] = { return_data[t_start][0],
                        return_data[t_start][1],
                        return_data[t_start][2] };
    Field B_start[] = { return_data[t_start][3],
                        return_data[t_start][4],
                        return_data[t_start][5] };

    Field E_final[] = { return_data[t_final][0],
                        return_data[t_final][1],
                        return_data[t_final][2] };
    Field B_final[] = { return_data[t_final][3],
                        return_data[t_final][4],
                        return_data[t_final][5] };

    std::cout << "reflection: " << 
        calc_reflection(E_start, B_start, E_final, B_final) 
        << std::endl;

    std::cout << "before: " << max_val_1 << std::endl;
    std::cout << "after: " << max_val_2 << std::endl;
#endif // __DEBUG_PML__

    return return_data;
}
