#pragma once

#include <utility>
#include <vector>
#include <functional>
#include <omp.h>
#include <cmath>
#include <fstream>

namespace FDTD_openmp {
    using Field = std::vector<double>;
    using Boundaries = std::pair<int, int>;
    using Function = std::function<int(int, int, int)>;
}
