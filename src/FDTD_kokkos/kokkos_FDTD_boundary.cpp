#include "kokkos_FDTD_boundary.h"

void FDTD_kokkos::applyPeriodicBoundary(int& i, int& j, int& k, int Ni, int Nj, int Nk)
{
    int i_isMinusOne = (i < 0);
	int j_isMinusOne = (j < 0);
	int k_isMinusOne = (k < 0);

	int i_isNi = (i == Ni);
	int j_isNj = (j == Nj);
	int k_isNk = (k == Nk);

	i = (Ni - 1) * i_isMinusOne + i *
		!(i_isMinusOne || i_isNi);
	j = (Nj - 1) * j_isMinusOne + j *
		!(j_isMinusOne || j_isNj);
	k = (Nk - 1) * k_isMinusOne + k *
		!(k_isMinusOne || k_isNk);
}
