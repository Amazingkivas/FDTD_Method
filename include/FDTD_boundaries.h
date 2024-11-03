#pragma once

#include <cmath>

#include "Structures.h"

namespace FDTD_boundaries
{
    void applyPeriodicBoundary(int& i, int& j, int& k, int Ni, int Nj, int Nk);
}
