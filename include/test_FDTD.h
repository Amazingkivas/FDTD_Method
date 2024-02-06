#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>
#include "FDTD.h"

enum class Axis {X, Y};

struct SelectedFields {
	Component selected_1;
	Component selected_2;

	Component calculated;
};

struct Parameters {
	double ax;
	double bx;

	double ay;
	double by;

	double dx;
	double dy;

	double time;
	int iterations;
};

struct Functions {
	std::function<double(double, double[2])>& init_function;
	std::function<double(double, double, double[2])>& true_function;
};

class Test_FDTD
{
private:
	double sign = 0.0;
	Axis axis;
	bool shifted;
	double max_abs_error = 0.0;

	void initial_filling(FDTD& _test, SelectedFields, double size_d, double size_wave[2],
		std::function<double(double, double[2])>& _init_function);

	void start_test(FDTD& _test, int _t);

	double get_shift(Component _field, double step);

	void get_max_abs_error(Field& this_field, Component field, double _size_d[2], double size_wave[2],
		std::function<double(double, double, double[2])>& _true_function, double _t);

public:
	Test_FDTD(FDTD& test, SelectedFields, Parameters, Functions, bool _shifted);

	double get_max_abs_error() { return max_abs_error; }
};
