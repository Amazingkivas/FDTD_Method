#pragma once

#include <vector>

namespace FDTDconst
{
	const double C = 1.0;  // light speed
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
		int iterations = static_cast<int>(static_cast<double>(period) / dt);;
		double period_x = static_cast<double>(m) * FDTDconst::C;
		double period_y = static_cast<double>(m) * FDTDconst::C;
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
