#pragma once

#include <Kokkos_Core.hpp>

#include <functional>
#include <vector>

namespace FDTD_kokkos
{
    using Device = Kokkos::DefaultExecutionSpace;
    using Field = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Aligned>>;  //Kokkos::SharedSpace>;
    using TimeField = std::vector<Field>;  //Kokkos::SharedSpace>;
    using Function = std::function<int(int, int, int)>;
    using InitFunction = std::function<double(double, double, double, double)>;
}

