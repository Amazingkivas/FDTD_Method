#include "test_FDTD.h" 

void Test_FDTD::initial_filling(FDTD& _test, Component field_1, Component field_2, double size_d, double size_wave[2],
	std::function<double(double, double[2])>& _init_function)
{
	if (axis == Axis::X)
	{
		double x = 0.0;
		double x_b = 0.0;
		for (int i = 0; i < _test.get_Ni(); x += size_d, ++i)
		{
			for (int j = 0; j < _test.get_Nj(); ++j)
			{
				_test.get_field(field_1)(i, j) = sign * _init_function(x, size_wave);
				if (shifted)
				{
					x_b = x + size_d / 2.0;
				}
				else x_b = x;
				_test.get_field(field_2)(i, j) = _init_function(x_b, size_wave);
			}
		}
	}
	else
	{
		double y = 0.0;
		double y_b = 0.0;
		for (int j = 0; j < _test.get_Nj(); y += size_d, ++j)
		{
			for (int i = 0; i < _test.get_Ni(); ++i)
			{
				_test.get_field(field_1)(i, j) = sign * _init_function(y, size_wave);
				if (shifted)
				{
					y_b = y + size_d / 2.0;
				}
				else y_b = y;
				_test.get_field(field_2)(i, j) = _init_function(y_b, size_wave);
			}
		}
	}
}

void Test_FDTD::start(FDTD& _test, double _t)
{
	if (shifted)
	{
		_test.shifted_update_field(_t);
	}
	else _test.update_field(_t);
}
double Test_FDTD::get_shift(Component _field, double step)
{
	if (static_cast<int>(_field) > static_cast<int>(Component::EZ))
	{
		sign = 1.0;
		return step / 2;
	}
	return 0.0;
}

void Test_FDTD::get_max_abs_error(Field& this_field, Component field, double _size_d[2], double size_wave[2],
	std::function<double(double, double, double[2])>& _true_function, double _t)
{
	double shift = 0.0;
	if (axis == Axis::X)
	{
		shift = get_shift(field, _size_d[0]);
		double extr_n = 0.0;
		double x = (shifted) ? shift : 0.0;
		int j = 0;
		for (int i = 0; i < this_field.get_Ni(); ++i, x += _size_d[0])
		{
			double this_n = fabs(sign * this_field(i, j) - _true_function(x, _t, size_wave));
			if (this_n > extr_n)
				extr_n = this_n;
		}
		max_abs_error = extr_n;
	}
	else
	{
		shift = get_shift(field, _size_d[1]);
		double extr_n = 0.0;
		double y = (shifted) ? shift : 0.0;
		int i = 0;
		for (int j = 0; j < this_field.get_Nj(); ++j, y += _size_d[1])
		{
			double this_n = fabs(sign * this_field(i, j) - _true_function(y, _t, size_wave));
			if (this_n > extr_n)
				extr_n = this_n;
		}
		max_abs_error = extr_n;
	}
}

Test_FDTD::Test_FDTD(FDTD& test, Component field_1, Component field_2, Component field_3,
	double size_x[2], double size_y[2], double size_d[2], double t,
	std::function<double(double, double[2])>& init_function,
	std::function<double(double, double, double[2])>& true_function, bool _shifted) : shifted(_shifted)
{
	if (field_1 == Component::EY && field_2 == Component::BZ || field_1 == Component::EZ && field_2 == Component::BX)
	{
		sign = 1.0;
	}
	else if (field_1 == Component::EX && field_2 == Component::BZ || field_1 == Component::EZ && field_2 == Component::BY)
	{
		sign = -1.0;
	}

	if (field_1 == Component::EY && field_2 == Component::BZ || field_1 == Component::EZ && field_2 == Component::BY)
	{
		axis = Axis::X;
		initial_filling(test, field_1, field_2, size_d[0], size_x, init_function);

		start(test, t);

		get_max_abs_error(test.get_field(field_3), field_3, size_d, size_x, true_function, t);
	}
	else if (field_1 == Component::EX && field_2 == Component::BZ || field_1 == Component::EZ && field_2 == Component::BX)
	{
		axis = Axis::Y;
		initial_filling(test, field_1, field_2, size_d[1], size_y, init_function);

		start(test, t);

		get_max_abs_error(test.get_field(field_3), field_3, size_d, size_y, true_function, t);
	}
	else
	{
		throw std::exception("ERROR: invalid selected fields");
	}
}
