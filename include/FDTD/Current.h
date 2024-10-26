#pragma once

#include <vector>

#include "Field.h"

class Current
{
private:
	std::vector<Field> J;
	int iterations;
	int Ni, Nj, Nk;
	double default_value = 0.0;
public:
	Current(int iters = 1, int _Ni = 1, int _Nj = 1, int _Nk = 1);

	Field& operator[] (int iteration);

	double operator() (int iteration, int i, int j, int k);
};
