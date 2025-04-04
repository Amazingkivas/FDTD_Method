#pragma once

#include "FDTD_kokkos.h"

#include <iostream>
#include <cmath>
#include <functional>

namespace FDTD_kokkos
{
	class Test_FDTD
	{
	private:
		Parameters parameters;
		double sign = 1.0;
		Axis axis = Axis::X;
		
		void set_sign(Component field_E, Component field_B);
		void set_axis(Component field_E, Component field_B);
		double get_shift(Component _field, double step);

	public:
		Test_FDTD(Parameters);

		void initial_filling(FDTD& _test, SelectedFields, int iters,
			std::function<double(double, double[2])>& init_function);

		double get_max_abs_error(Field& this_field, Component field,
			std::function<double(double, double, double[2])>& true_function, double time);
	};

	Test_FDTD::Test_FDTD(Parameters _parameters) : parameters(_parameters) {}

	void Test_FDTD::initial_filling(FDTD& _test, SelectedFields fields, int iters,
		std::function<double(double, double[2])>& init_function)
	{
		set_axis(fields.selected_E, fields.selected_B);
		set_sign(fields.selected_E, fields.selected_B);
		if (axis == Axis::X)
		{
			double x_b = parameters.dx / 2.0;
			double size_x[2] = { parameters.ax, parameters.bx };
			for (int i = 0; i < parameters.Ni; i++)
			{
				double x = static_cast<double>(i) * parameters.dx;
				for (int j = 0; j < parameters.Nj; j++)
				{
					for (int k = 0; k < parameters.Nk; k++)
					{
						int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
						_test.get_field(fields.selected_E)(index) = sign * init_function(x, size_x);
						_test.get_field(fields.selected_B)(index) = init_function(x_b + x, size_x);
					}
				}
			}
		}
		else if (axis == Axis::Y)
		{
			double y_b = parameters.dy / 2.0;
			double size_y[2] = { parameters.ay, parameters.by };
			for (int j = 0; j < parameters.Nj; j++)
			{
				double y = static_cast<double>(j) * parameters.dy;
				for (int k = 0; k < parameters.Nk; k++)
				{
					for (int i = 0; i < parameters.Ni; i++)
					{
						int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
						_test.get_field(fields.selected_E)(index) = sign * init_function(y, size_y);
						_test.get_field(fields.selected_B)(index) = init_function(y_b + y, size_y);
					}
				}
			}
		}
		else
		{
			double z_b = parameters.dz / 2.0;
			double size_z[2] = { parameters.az, parameters.bz };
			for (int k = 0; k < parameters.Nk; k++)
			{
				double z = static_cast<double>(k) * parameters.dz;
				for (int i = 0; i < parameters.Ni; i++)
				{
					for (int j = 0; j < parameters.Nj; j++)
					{
						int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
						_test.get_field(fields.selected_E)(index) = sign * init_function(z, size_z);
						_test.get_field(fields.selected_B)(index) = init_function(z_b + z, size_z);
					}
				}
			}
		}
	}

	void Test_FDTD::set_sign(Component field_E, Component field_B)
	{
		if (field_E == Component::EX && field_B == Component::BZ ||
			field_E == Component::EZ && field_B == Component::BY ||
			field_E == Component::EY && field_B == Component::BX)
		{
			sign = -1.0;
		}
		else if (field_E == Component::EY && field_B == Component::BZ ||
				field_E == Component::EZ && field_B == Component::BX ||
				field_E == Component::EX && field_B == Component::BY)
		{
			sign = 1.0;
		}
		else throw std::logic_error("ERROR: invalid selected fields");
	}
	void Test_FDTD::set_axis(Component field_E, Component field_B)
	{
		if (field_E == Component::EY && field_B == Component::BZ ||
			field_E == Component::EZ && field_B == Component::BY)
		{
			axis = Axis::X;
		}
		else if (field_E == Component::EX && field_B == Component::BZ ||
				field_E == Component::EZ && field_B == Component::BX)
		{
			axis = Axis::Y;
		}
		else if (field_E == Component::EX && field_B == Component::BY ||
				field_E == Component::EY && field_B == Component::BX)
		{
			axis = Axis::Z;
		}
		else throw std::logic_error("ERROR: invalid selected fields");
	}
	double Test_FDTD::get_shift(Component _field, double step)
	{
		if (static_cast<int>(_field) > static_cast<int>(Component::EZ))
		{
			sign = 1.0;
			return step / 2.0;
		}
		return 0.0;
	}

	double Test_FDTD::get_max_abs_error(Field& this_field, Component field,
		std::function<double(double, double, double[2])>& true_function, double time)
	{
		double this_error = 0.0;
		double max_abs_error = 0.0;
		if (axis == Axis::X)
		{
			double x = get_shift(field, parameters.dx);
			double size_x[2] = { parameters.ax, parameters.bx };
			int j = 0;
			int k = 0;
			for (int i = 0; i < parameters.Ni; ++i, x += parameters.dx)
			{
				int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
				this_error = fabs(sign * this_field(index) - true_function(x, time, size_x));
				if (this_error > max_abs_error)
					max_abs_error = this_error;
			}
		}
		else if (axis == Axis::Y)
		{
			double y = get_shift(field, parameters.dy);
			double size_y[2] = { parameters.ay, parameters.by };
			int i = 0;
			int k = 0;
			for (int j = 0; j < parameters.Nj; ++j, y += parameters.dy)
			{
				int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
				this_error = fabs(sign * this_field(index) - true_function(y, time, size_y));
				if (this_error > max_abs_error)
					max_abs_error = this_error;
			}
		}
		else
		{
			double z = get_shift(field, parameters.dz);
			double size_z[2] = { parameters.az, parameters.bz };
			int i = 0;
			int j = 0;
			for (int k = 0; k < parameters.Nk; ++k, z += parameters.dz)
			{
				int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
				this_error = fabs(sign * this_field(index) - true_function(z, time, size_z));
				if (this_error > max_abs_error)
					max_abs_error = this_error;
			}
		}
		return max_abs_error;
	}

}