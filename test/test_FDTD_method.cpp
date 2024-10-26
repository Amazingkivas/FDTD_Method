#define _USE_MATH_DEFINES

#include "test_FDTD.h"

#include <gtest/gtest.h>

const double default_time = 5e-13;

std::function<double(double, double[2])> initial_func = [](double x, double size[2])
{
	return sin(2.0 * M_PI * (x - size[0]) / (size[1] - size[0]));
};
std::function<double(double, double, double[2])> true_func = [](double x, double t, double size[2])
{
	return sin(2.0 * M_PI * (x - size[0] - FDTDconst::C * t) / (size[1] - size[0]));
};

double run_test(Component test_field, SelectedFields current_fields, int Ni, int Nj, int Nk) {
    //---------------------
    Parameters params_1{
        Ni,
        Nj,
        Nk,
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0,
        (1 - 0) / static_cast<double>(Ni),
        (2 - 0) / static_cast<double>(Nj),
        (3 - 0) / static_cast<double>(Nk)
    };
	int iters_1 = 16;
    double dt_1 = default_time / static_cast<double>(iters_1);
    FDTD method_1(params_1, dt_1, 0.0);

    Test_FDTD test_1(params_1);
    test_1.initial_filling(method_1, current_fields, iters_1, initial_func);
    method_1.update_fields(iters_1);

    double err_1 = test_1.get_max_abs_error(method_1.get_field(test_field), test_field, true_func, default_time);
    //---------------------

    //---------------------
    Parameters params_2{
        Ni * 2,
        Nj * 2,
        Nk * 2,
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0,
        (1 - 0) / static_cast<double>(Ni * 2),
        (2 - 0) / static_cast<double>(Nj * 2),
        (3 - 0) / static_cast<double>(Nk * 2)
    };
	int iters_2 = iters_1 * 2;
    double dt_2 = default_time / static_cast<double>(iters_2);
    FDTD method_2(params_2, dt_2, 0.0);

    Test_FDTD test_2(params_2);
    test_2.initial_filling(method_2, current_fields, iters_2, initial_func);
    method_2.update_fields(iters_2);

    double err_2 = test_2.get_max_abs_error(method_2.get_field(test_field), test_field, true_func, default_time);
    //---------------------

    return err_1 / err_2;
}

TEST(Convergence, x_axis_EY) 
{
    SelectedFields current_fields
	{
		Component::EY,
		Component::BZ
	};
    Component test_field = Component::EY;
    ASSERT_NEAR(run_test(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence, x_axis_BZ) 
{
    SelectedFields current_fields
	{
		Component::EY, 
		Component::BZ
	};
    Component test_field = Component::BZ;
    ASSERT_NEAR(run_test(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}



TEST(Convergence, x_axis_EZ) 
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BY
	};
	Component test_field = Component::EZ;
	ASSERT_NEAR(run_test(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence, x_axis_BY)
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BY
	};
	Component test_field = Component::BY;
	ASSERT_NEAR(run_test(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence, y_axis_EX)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BZ
	};
	Component test_field = Component::EX;
	ASSERT_NEAR(run_test(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence, y_axis_BZ)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BZ
	};
	Component test_field = Component::BZ;
	ASSERT_NEAR(run_test(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence, y_axis_EZ)
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BX
	};
	Component test_field = Component::EZ;
	ASSERT_NEAR(run_test(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence, y_axis_BX)
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BX
	};
	Component test_field = Component::BX;
	ASSERT_NEAR(run_test(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence, z_axis_EX)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BY
	};
	Component test_field = Component::EX;
	ASSERT_NEAR(run_test(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}

TEST(Convergence, z_axis_BY)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BY
	};
	Component test_field = Component::BY;
	ASSERT_NEAR(run_test(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}

TEST(Convergence, z_axis_EY)
{
	SelectedFields current_fields
	{
		Component::EY,
		Component::BX
	};
	Component test_field = Component::EY;
	ASSERT_NEAR(run_test(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}

TEST(Convergence, z_axis_BX)
{
	SelectedFields current_fields
	{
		Component::EY,
		Component::BX
	};
	Component test_field = Component::BX;
	ASSERT_NEAR(run_test(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}
