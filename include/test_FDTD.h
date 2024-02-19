#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>
#include "FDTD.h"
#include "Structures.h"

enum class Axis {X, Y, Z};

class Test_FDTD
{
private:
	double sign = 1.0;
	bool shifted;

	Axis axis;
	SelectedFields fields;
	Parameters parameters;
	Functions functions;

	void initial_filling(FDTD& _test, SelectedFields, double size_d, double size_wave[2],
		std::function<double(double, double[2])>& _init_function);

	double get_shift(Component _field, double step);
	void set_sign(Component field_E, Component field_B);

	double max_abs_error = 0.0;
	void get_max_abs_error(Field& this_field, Component field, double _size_d[3], double size_wave[2],
		std::function<double(double, double, double[2])>& _true_function, double _t);

public:
	Test_FDTD(SelectedFields, Parameters, Functions, bool _shifted = true);

	void run_test(FDTD& test);

	double get_max_abs_error() { return max_abs_error; }
};
