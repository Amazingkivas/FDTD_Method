#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>

#include "FDTD.h"

class Test_FDTD
{
private:
	Parameters parameters;
	bool shifted;
	double sign = 1.0;
	double shift = 0.0;
	Axis axis;
	
	void set_sign(Component field_E, Component field_B);
	void set_axis(Component field_E, Component field_B);
	double get_shift(Component _field, double step);

public:
	Test_FDTD(Parameters, bool _shifted = true);

	void initial_filling(FDTD& _test, SelectedFields, 
		std::function<double(double, double[2])>& init_function);

	double get_max_abs_error(Field& this_field, Component field, 
		std::function<double(double, double, double[2])>& true_function, double time);
};
