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
		int Ni;
		int Nj;
		int Nk;

		double ax;
		double bx;

		double ay;
		double by;

		double az;
		double bz;

		double dx = (bx - ax) / static_cast<double>(Ni);
		double dy = (by - ay) / static_cast<double>(Nj);
		double dz = (bz - az) / static_cast<double>(Nk);
	};
}
