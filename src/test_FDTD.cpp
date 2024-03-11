#include "test_FDTD.h" 

Test_FDTD::Test_FDTD(Parameters _parameters) : parameters(_parameters) {}

void Test_FDTD::initiialize_current(FDTD& _test, CurrentParameters cParams, int iters,
	std::function<double(double, double, double, double)>& init_function)
{
	double Tx = cParams.period_x;
	double Ty = cParams.period_y;
	double Tz = cParams.period_z;
	double T = cParams.period;

	int start_i = std::floor((-Tx / 4.0 - parameters.ax) / parameters.dx);
	int start_j = std::floor((-Ty / 4.0 - parameters.ay) / parameters.dy);
	int start_k = std::floor((-Tz / 4.0 - parameters.ay) / parameters.dy);

	int max_i = std::floor((Tx / 4.0 - parameters.ax) / parameters.dx);
	int max_j = std::floor((Ty / 4.0 - parameters.ay) / parameters.dy);
	int max_k = std::floor((Tz / 4.0 - parameters.az) / parameters.dz);

	for (int n = 0; n < std::max(iters, cParams.iterations); n++)
	{
		_test.get_current(Component::JX).push_back(Field(parameters.Ni, parameters.Nj, parameters.Nk));
		_test.get_current(Component::JY).push_back(Field(parameters.Ni, parameters.Nj, parameters.Nk));
		_test.get_current(Component::JZ).push_back(Field(parameters.Ni, parameters.Nj, parameters.Nk));
	}
	for (int iter = 1; iter <= cParams.iterations; iter++)
	{
		Field J(parameters.Ni, parameters.Nj, parameters.Nk);

		for (int i = start_i; i <= max_i; i++)
		{
			for (int j = start_j; j <= max_j; j++)
			{
				for (int k = start_k; k <= max_k; k++)
				{
					J(i, j, k) = init_function(static_cast<double>(i) * parameters.dx,
						static_cast<double>(j) * parameters.dy,
						static_cast<double>(k) * parameters.dz,
						static_cast<double>(iter) * cParams.dt);
				}
			}
		}
		_test.get_current(Component::JX)[iter - 1] = J;
		_test.get_current(Component::JY)[iter - 1] = J;
		_test.get_current(Component::JZ)[iter - 1] = J;
	}
}

void Test_FDTD::initial_filling(FDTD& _test, SelectedFields fields, int iters,
	std::function<double(double, double[2])>& init_function)
{
	for (int n = 0; n < iters; n++)
	{
		_test.get_current(Component::JX).push_back(Field(parameters.Ni, parameters.Nj, parameters.Nk));
		_test.get_current(Component::JY).push_back(Field(parameters.Ni, parameters.Nj, parameters.Nk));
		_test.get_current(Component::JZ).push_back(Field(parameters.Ni, parameters.Nj, parameters.Nk));
	}

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
					_test.get_field(fields.selected_E)(i, j, k) = sign * init_function(x, size_x);
					_test.get_field(fields.selected_B)(i, j, k) = init_function(x_b + x, size_x);
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
					_test.get_field(fields.selected_E)(i, j, k) = sign * init_function(y, size_y);
					_test.get_field(fields.selected_B)(i, j, k) = init_function(y_b + y, size_y);
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
					_test.get_field(fields.selected_E)(i, j, k) = sign * init_function(z, size_z);
					_test.get_field(fields.selected_B)(i, j, k) = init_function(z_b + z, size_z);
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
	else throw std::exception("ERROR: invalid selected fields");
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
	else throw std::exception("ERROR: invalid selected fields");
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
			this_error = fabs(sign * this_field(i, j, k) - true_function(x, time, size_x));
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
			this_error = fabs(sign * this_field(i, j, k) - true_function(y, time, size_y));
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
			this_error = fabs(sign * this_field(i, j, k) - true_function(z, time, size_z));
			if (this_error > max_abs_error)
				max_abs_error = this_error;
		}
	}
	return max_abs_error;
}
