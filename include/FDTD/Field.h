#pragma once

#include <vector>

class Field
{
private:
	int Ni, Nj, Nk;
	std::vector<double> field;

public:
	Field(const int _Ni = 1, const int _Nj = 1, const int _Nk = 1);
	Field& operator= (const Field& other);

	double& operator() (int i, int j, int k);

	int get_Ni() { return Ni; }
	int get_Nj() { return Nj; }
	int get_Nk() { return Nk; }
};
