#pragma once

#include <vector>

namespace FDTDconst
{
	const double C = 3e10;
	const double R = 0.01;
	const double EPS0 = 1.60217656535e-12;
	const double PI = 3.1415926535;
	const double MU0 = 1e6 * EPS0;
	const double N = 4.0;
}

namespace FDTDstruct
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
		double dt;
		int iterations;
		double period_x = static_cast<double>(m) * FDTDconst::C;
		double period_y = static_cast<double>(m) * FDTDconst::C;
		double period_z = static_cast<double>(m) * FDTDconst::C;
	};

	struct PMLparameters {
		double sigma_low_xy;
		double sigma_low_yz;
		double sigma_low_zx;
		double sigma_up_xy;
		double sigma_up_yz;
		double sigma_up_zx;
	};

	struct Parameters {
		int Ni;
		int Nj;
		int Nk;

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
