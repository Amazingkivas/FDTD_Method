#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

#include <utility>
#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

namespace FDTD_kokkos
{
    using Device = Kokkos::DefaultExecutionSpace;
    using Field = Kokkos::View<double*, Device>;  //Kokkos::SharedSpace>;
    using Boundaries = std::pair<int, int>;
    using Function = std::function<int(int, int, int)>;
}
