#include "Field.h"

Field::Field(const int _Ni, const int _Nj, const int _Nk)
	: Ni(_Ni), Nj(_Nj), Nk(_Nk)
{
	int size = Ni * Nj * Nk;
	field = std::vector<double>(size, 0.0);
}

Field& Field::operator= (const Field& other)
{
	if (this != &other)
	{
		field = other.field;
		Ni = other.Ni;
		Nj = other.Nj;
		Nk = other.Nk;
	}
	return *this;
}

double& Field::operator() (const int& i, const int& j, const int& k)
{
	return field[i + j * Ni + k * Ni * Nj];
}
