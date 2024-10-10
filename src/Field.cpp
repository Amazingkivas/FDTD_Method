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

double& Field::operator() (int i, int j, int k)
{
	int i_isMinusOne = (i == -1);
	int j_isMinusOne = (j == -1);
	int k_isMinusOne = (k == -1);
	int i_isNi = (i == Ni);
	int j_isNj = (j == Nj);
	int k_isNk = (k == Nk);

	int truly_i = (Ni - 1) * i_isMinusOne + i *
		!(i_isMinusOne || i_isNi);
	int truly_j = (Nj - 1) * j_isMinusOne + j *
		!(j_isMinusOne || j_isNj);
	int truly_k = (Nk - 1) * k_isMinusOne + k *
		!(k_isMinusOne || k_isNk);

	int index = truly_i + truly_j * Ni + truly_k * Ni * Nj;
	return field[index];
}

