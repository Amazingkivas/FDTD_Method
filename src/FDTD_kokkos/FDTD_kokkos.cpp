#include "FDTD_kokkos.h"

using namespace FDTD_kokkos;

FDTD::FDTD(Parameters _parameters, double _dt, double _pml_percent, int time_, CurrentParameters _Cpar,
    std::function<double(double, double, double, double)> init_function) :
    parameters(_parameters), cParams(_Cpar), dt(_dt), pml_percent(_pml_percent), time(time_), init_func(init_function)

{
    if (parameters.Ni <= 0 ||
        parameters.Nj <= 0 ||
        parameters.Nk <= 0 ||
        dt <= 0)
    {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    Jx = TimeField("Jx", time, parameters.Ni, parameters.Nj, parameters.Nk);
    Jy = TimeField("Jy", time, parameters.Ni, parameters.Nj, parameters.Nk);
    Jz = TimeField("Jz", time, parameters.Ni, parameters.Nj, parameters.Nk);

    Ex = Field("Ex", parameters.Ni, parameters.Nj, parameters.Nk);
    Ey = Field("Ey", parameters.Ni, parameters.Nj, parameters.Nk);
    Ez = Field("Ez", parameters.Ni, parameters.Nj, parameters.Nk);
    Bx = Field("Bx", parameters.Ni, parameters.Nj, parameters.Nk);
    By = Field("By", parameters.Ni, parameters.Nj, parameters.Nk);
    Bz = Field("Bz", parameters.Ni, parameters.Nj, parameters.Nk);

    Exy = Field("Exy", parameters.Ni, parameters.Nj, parameters.Nk);
    Exz = Field("Exz", parameters.Ni, parameters.Nj, parameters.Nk);
    Eyx = Field("Eyx", parameters.Ni, parameters.Nj, parameters.Nk);
    Eyz = Field("Eyz", parameters.Ni, parameters.Nj, parameters.Nk);
    Ezx = Field("Ezx", parameters.Ni, parameters.Nj, parameters.Nk);
    Ezy = Field("Ezy", parameters.Ni, parameters.Nj, parameters.Nk);

    Bxy = Field("Bxy", parameters.Ni, parameters.Nj, parameters.Nk);
    Bxz = Field("Bxz", parameters.Ni, parameters.Nj, parameters.Nk);
    Byx = Field("Byx", parameters.Ni, parameters.Nj, parameters.Nk);
    Byz = Field("Byz", parameters.Ni, parameters.Nj, parameters.Nk);
    Bzx = Field("Bzx", parameters.Ni, parameters.Nj, parameters.Nk);
    Bzy = Field("Bzy", parameters.Ni, parameters.Nj, parameters.Nk);

    EsigmaX = Field("EsigmaX", parameters.Ni, parameters.Nj, parameters.Nk);
    EsigmaY = Field("EsigmaY", parameters.Ni, parameters.Nj, parameters.Nk);
    EsigmaZ = Field("EsigmaZ", parameters.Ni, parameters.Nj, parameters.Nk);

    BsigmaX = Field("BsigmaX", parameters.Ni, parameters.Nj, parameters.Nk);
    BsigmaY = Field("BsigmaY", parameters.Ni, parameters.Nj, parameters.Nk);
    BsigmaZ = Field("BsigmaZ", parameters.Ni, parameters.Nj, parameters.Nk);

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

void FDTD::update_B_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2])
{
    ComputeB_PML_FieldFunctor::apply(Ex, Ey, Ez, Bxy, Byx, Bzy, Byz, Bzx, Bxz,
                                   Bx, By, Bz, BsigmaX, BsigmaY, BsigmaZ,
                                   dt, parameters.dx, parameters.dy, parameters.dz,
                                   bounds_i, bounds_j, bounds_k,
                                   parameters.Ni, parameters.Nj, parameters.Nk);
}
void FDTD::update_E_PML(int bounds_i[2], int bounds_j[2], int bounds_k[2])
{
    ComputeE_PML_FieldFunctor::apply(Ex, Ey, Ez, Exy, Eyx, Ezy, Eyz, Ezx, Exz,
                                   Bx, By, Bz, EsigmaX, EsigmaY, EsigmaZ,
                                   dt, parameters.dx, parameters.dy, parameters.dz,
                                   bounds_i, bounds_j, bounds_k,
                                   parameters.Ni, parameters.Nj, parameters.Nk);
}

void FDTD::update_fields(bool write_result, Axis write_axis, std::string base_path)
{
    if (time < 0)
    {
        throw std::invalid_argument("ERROR: Invalid update field argument");
    }

    int cur_time;

    if (init_func)
    {
        cur_time = cParams.iterations;

        double Tx = cParams.period_x;
        double Ty = cParams.period_y;
        double Tz = cParams.period_z;
        
        int start_i = static_cast<int>(floor((-Tx / 4.0 - parameters.ax) / parameters.dx));
        int start_j = static_cast<int>(floor((-Ty / 4.0 - parameters.ay) / parameters.dy));
        int start_k = static_cast<int>(floor((-Tz / 4.0 - parameters.az) / parameters.dz));

        int max_i = static_cast<int>(floor((Tx / 4.0 - parameters.ax) / parameters.dx));
        int max_j = static_cast<int>(floor((Ty / 4.0 - parameters.ay) / parameters.dy));
        int max_k = static_cast<int>(floor((Tz / 4.0 - parameters.az) / parameters.dz));

        int size_i_cur[2] = { start_i, max_i };
        int size_j_cur[2] = { start_j, max_j };
        int size_k_cur[2] = { start_k, max_k };

        InitializeCurrentFunctor::apply(Jx, cParams, parameters, init_func, cur_time,
                                        size_i_cur, size_j_cur, size_k_cur);
        InitializeCurrentFunctor::apply(Jy, cParams, parameters, init_func, cur_time,
                                        size_i_cur, size_j_cur, size_k_cur);
        InitializeCurrentFunctor::apply(Jz, cParams, parameters, init_func, cur_time,
                                        size_i_cur, size_j_cur, size_k_cur);
    }
    else
    {
        cur_time = 0;
    }

    if (pml_percent == 0.0)
    {
        int size_i_main[2] = { 0, parameters.Ni };
        int size_j_main[2] = { 0, parameters.Nj };
        int size_k_main[2] = { 0, parameters.Nk };
        for (int t = 0; t < time; t++)
        {
            std::cout << "Iteration: " << t + 1 << std::endl;

            ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt, 
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main,
            parameters.Ni, parameters.Nj, parameters.Nk);

            ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
             Jx, Jy, Jz, dt, 
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main, t, cur_time,
            parameters.Ni, parameters.Nj, parameters.Nk);

            ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main,
            parameters.Ni, parameters.Nj, parameters.Nk);

            auto Ex_host = Kokkos::create_mirror_view(Ex);
            auto Ey_host = Kokkos::create_mirror_view(Ey);
            auto Ez_host = Kokkos::create_mirror_view(Ez);
            auto Bx_host = Kokkos::create_mirror_view(Bx);
            auto By_host = Kokkos::create_mirror_view(By);
            auto Bz_host = Kokkos::create_mirror_view(Bz);

            Kokkos::deep_copy(Ex_host, Ex);
            Kokkos::deep_copy(Ey_host, Ey);
            Kokkos::deep_copy(Ez_host, Ez);
            Kokkos::deep_copy(Bx_host, Bx);
            Kokkos::deep_copy(By_host, By);
            Kokkos::deep_copy(Bz_host, Bz);

            std::vector<Field> return_data{ Ex_host, Ey_host, Ez_host, Bx_host, By_host, Bz_host };
            if (write_result)
                write_spherical(return_data, write_axis, base_path, t);
        }
        return;
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

    // Calculation of maximum permittivity and permeability
    double SGm_x = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<double>(pml_size_i) * parameters.dx);
    double SGm_y = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<double>(pml_size_j) * parameters.dy);
    double SGm_z = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<double>(pml_size_k) * parameters.dz);

    // Calculation of permittivity and permeability in the cells
    ComputeSigmaFunctor::apply(EsigmaZ, BsigmaZ, SGm_z, calc_distant_k_low, pml_size_k, dt,
        size_i_solid, size_j_solid, size_xy_lower_k_pml);
    ComputeSigmaFunctor::apply(EsigmaY, BsigmaY, SGm_y, calc_distant_j_low, pml_size_j, dt,
        size_i_solid, size_zx_lower_j_pml, size_k_solid);
    ComputeSigmaFunctor::apply(EsigmaX, BsigmaX, SGm_x, calc_distant_i_low, pml_size_i, dt,
        size_yz_lower_i_pml, size_j_solid, size_k_solid);

    ComputeSigmaFunctor::apply(EsigmaZ, BsigmaZ, SGm_z, calc_distant_k_up, pml_size_k, dt,
        size_i_solid, size_j_solid, size_xy_upper_k_pml);
    ComputeSigmaFunctor::apply(EsigmaY, BsigmaY, SGm_y, calc_distant_j_up, pml_size_j, dt,
        size_i_solid, size_zx_upper_j_pml, size_k_solid);
    ComputeSigmaFunctor::apply(EsigmaX, BsigmaX, SGm_x, calc_distant_i_up, pml_size_i, dt,
        size_yz_upper_i_pml, size_j_solid, size_k_solid);

    for (int t = 0; t < time; t++)
    {
        std::cout << "Iteration: " << t + 1 << std::endl;

        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt, 
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main,
            parameters.Ni, parameters.Nj, parameters.Nk);

        update_B_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
        update_B_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
        update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

        update_B_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
        update_B_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
        update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

        ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
            Jx, Jy, Jz, dt, 
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main, t, cur_time,
            parameters.Ni, parameters.Nj, parameters.Nk);

        update_E_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
        update_E_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
        update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

        update_E_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
        update_E_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
        update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

        ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, dt,
            parameters.dx, parameters.dy, parameters.dz,
            size_i_main, size_j_main, size_k_main,
            parameters.Ni, parameters.Nj, parameters.Nk);

        auto Ex_host = Kokkos::create_mirror_view(Ex);
        auto Ey_host = Kokkos::create_mirror_view(Ey);
        auto Ez_host = Kokkos::create_mirror_view(Ez);
        auto Bx_host = Kokkos::create_mirror_view(Bx);
        auto By_host = Kokkos::create_mirror_view(By);
        auto Bz_host = Kokkos::create_mirror_view(Bz);

        Kokkos::deep_copy(Ex_host, Ex);
        Kokkos::deep_copy(Ey_host, Ey);
        Kokkos::deep_copy(Ez_host, Ez);
        Kokkos::deep_copy(Bx_host, Bx);
        Kokkos::deep_copy(By_host, By);
        Kokkos::deep_copy(Bz_host, Bz);

        std::vector<Field> return_data{ Ex_host, Ey_host, Ez_host, Bx_host, By_host, Bz_host };
        if (write_result)
            write_spherical(return_data, write_axis, base_path, t);
    }
}

void FDTD::write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it)
{
    std::ofstream test_fout;
    switch (axis)
    {
    case Axis::X:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int k = 0; k < parameters.Nk; ++k)
            {
                for (int j = 0; j < parameters.Nj; ++j)
                {
                    test_fout << field(parameters.Ni / 2, j, k);
                    if (j == parameters.Nj - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    case Axis::Y:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int i = 0; i < parameters.Nj; ++i)
            {
                for (int k = 0; k < parameters.Nk; ++k)
                {
                    test_fout << field(i, parameters.Nj / 2, k);
                    if (k == parameters.Nk - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    case Axis::Z:
    {
        for (int c = static_cast<int>(Component::EX); c <= static_cast<int>(Component::BZ); ++c)
        {
            test_fout.open(base_path + "OutFiles_" + std::to_string(c + 1) + "/" + std::to_string(it) + ".csv");
            Field field = fields[c];
            for (int i = 0; i < parameters.Nj; ++i)
            {
                for (int j = 0; j < parameters.Nk; ++j)
                {
                    test_fout << field(i, j, parameters.Nk / 2);
                    if (j == parameters.Nk - 1)
                    {
                        test_fout << std::endl;
                    }
                    else
                    {
                        test_fout << ";";
                    }
                }
            }
            test_fout.close();
        }
        break;
    }
    default: throw std::logic_error("ERROR: Invalid axis");
    }
}

