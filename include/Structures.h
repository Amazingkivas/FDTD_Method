#pragma once

#include "FDTD.h"

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

	double az;
	double bz;

	double dx;
	double dy;
	double dz;

	double time;
	int iterations;
};

struct Functions {
	std::function<double(double, double[2])>* init_function = nullptr;
	std::function<double(double, double, double[2])>* true_function = nullptr;
};
