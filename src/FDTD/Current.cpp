#include "Current.h"

#include <iostream>

Current::Current(int iters, int _Ni, int _Nj, int _Nk) 
	: iterations(iters), Ni(_Ni), Nj(_Nj), Nk(_Nk)
{
	J = std::vector<Field>(iterations, Field(Ni, Nj, Nk));
}

Field& Current::operator[] (int iteration)
{
	if (iteration < iterations)
	{
		return J[iteration];
	}
	else
	{
		throw std::invalid_argument("Exceeding the specified number of iterations");
	}
}

double Current::operator() (int iteration, int i, int j, int k)
{
	if (iteration < iterations)
	{
		return J[iteration](i, j, k);
	}
	else
	{
		return default_value;
	}
}
