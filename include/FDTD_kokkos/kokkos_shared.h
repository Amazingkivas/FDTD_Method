#pragma once

#include <Kokkos_Core.hpp>

#include <functional>

namespace FDTD_kokkos
{
    using Device = Kokkos::DefaultExecutionSpace;
    using Field = Kokkos::View<double***, Device>;
    using TimeField = Kokkos::View<double****, Device>;
    using Function = std::function<int(int, int, int)>;
}
