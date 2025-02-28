#pragma once

#include <vector>
#include <functional>

namespace FDTD_openmp {
    using Field = std::vector<std::vector<std::vector<double>>>;
    using TimeField = std::vector<std::vector<std::vector<std::vector<double>>>>;
    using Function = std::function<int(int, int, int)>;
}
