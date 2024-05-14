#pragma once

#include <vector>

namespace FDTDconst
{
	const double C = 3e10;
	const double R = 1e-12;
	const double EPS0 = 1.0;
	const double MU0 = EPS0;
	const double N = 4.0;
	const double PI = 3.14159265358;
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
