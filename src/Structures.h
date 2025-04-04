#pragma once

namespace FDTD_const
{
	const double C = 3e10;
	const double R = 1e-12;
	const double EPS0 = 1.0;
	const double MU0 = EPS0;
	const double N = 4.0;
	const double PI = 3.14159265358;
}

namespace FDTD_struct
{
	enum class Component { EX, EY, EZ, BX, BY, BZ, JX, JY, JZ };
	enum class Axis { X, Y, Z };

	struct SelectedFields {
		Component selected_E;
		Component selected_B;
	};

	struct CurrentParameters {
		int period;
		int m;
		int iterations;
		double period_x = static_cast<double>(m) * FDTD_const::C;
		double period_y = static_cast<double>(m) * FDTD_const::C;
		double period_z = static_cast<double>(m) * FDTD_const::C;
	};

	struct Parameters {
		int Ni;
		int Nj;
		int Nk;

		double dt;

		double ax;
		double bx;
		double ay;
		double by;
		double az;
		double bz;

		double dx;
		double dy;
		double dz;
	};
}
