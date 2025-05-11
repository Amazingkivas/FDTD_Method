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

    int size = (parameters.Ni) * (parameters.Nj) * (parameters.Nk);

    Jx = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Jx"), size);
    Ex = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Ex"), size);
    Ey = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Ey"), size);
    Ez = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Ez"), size);
    Bx = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Bx"), size);
    By = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "By"), size);
    Bz = Field(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Bz"), size);

    Kokkos::parallel_for("FirstTouch", Kokkos::RangePolicy<>(0, size), KOKKOS_LAMBDA(int i) {
        Ex(i) = 0.0;
        Ey(i) = 0.0;
        Ez(i) = 0.0;
        Bx(i) = 0.0;
        By(i) = 0.0;
        Bz(i) = 0.0;
        Jx(i) = 0.0;
    });
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
    }
    else
    {
        cur_time = 0;
    }

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

    int Ni = parameters.Ni;
    int Nj = parameters.Nj;
    int Nk = parameters.Nk;

    double coef_E = FDTD_const::C * dt;
    double coef_B = FDTD_const::C * dt / 2.0;
    double current_coef = 4.0 * FDTD_const::PI * dt;
    if (pml_percent == 0.0)
    {
        int size_i_main[2] = { 0, parameters.Ni };
        int size_j_main[2] = { 0, parameters.Nj };
        int size_k_main[2] = { 0, parameters.Nk };
        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < time; t++)
        {

            for (int i = start_i; i < max_i; ++i)
            {
                for (int j = start_j; j < max_j; ++j)
                {
                    for (int k = start_k; k < max_k; ++k)
                    {
                        int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
                        Jx[index] = init_func(static_cast<double>(i) * parameters.dx,
                                        static_cast<double>(j) * parameters.dy,
                                        static_cast<double>(k) * parameters.dz,
                                        static_cast<double>(t + 1) * cParams.dt);
                    }
                }
            }

            ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz, 
            size_i_main, size_j_main, size_k_main,
            Ni, Nj, Nk,
            coef_B / parameters.dx, coef_B / parameters.dy, coef_B / parameters.dz);

            ComputeE_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
             Jx, Jx, Jx, current_coef,
            size_i_main, size_j_main, size_k_main, t, cur_time,
            Ni, Nj, Nk,
            coef_E / parameters.dx, coef_E / parameters.dy, coef_E / parameters.dz);

            ComputeB_FieldFunctor::apply(Ex, Ey, Ez, Bx, By, Bz,
            size_i_main, size_j_main, size_k_main,
            Ni, Nj, Nk,
            coef_B / parameters.dx, coef_B / parameters.dy, coef_B / parameters.dz);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "BETTER Execution time: " << elapsed.count() << " s" << std::endl;
        return;
    }
}




