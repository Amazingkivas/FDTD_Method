#pragma once

namespace FDTDstruct
{
	enum class Component { EX, EY, EZ, BX, BY, BZ };

	enum class Axis { X, Y, Z };

	struct SelectedFields {
		Component selected_E;
		Component selected_B;
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
	};
}
