#pragma once

#include <iostream>
#include <cmath>
#include <functional>

#include "FDTD.h"

namespace FDTD_openmp {

class Test_FDTD
{
private:
	Parameters parameters;
	double sign = 1.0;
	Axis axis = Axis::X;
	
	void set_sign(Component field_E, Component field_B);
	void set_axis(Component field_E, Component field_B);
	double get_shift(Component _field, double step);

public:
	Test_FDTD(Parameters);

	void initial_filling(FDTD& _test, SelectedFields, int iters,
		std::function<double(double, double[2])>& init_function);

	double get_max_abs_error(Field& this_field, Component field,
		std::function<double(double, double, double[2])>& true_function, double time);
};

}
