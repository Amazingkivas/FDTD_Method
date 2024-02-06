#define _USE_MATH_DEFINES
//#define __DEBUG__

#include <iostream>
#include <iomanip>
#include <fstream>

#include "test_FDTD.h"
#include "Writer.h"

#ifndef __DEBUG__
int main(int argc, char* argv[])
#else
int main()
#endif
{
    std::vector<double> numbers;
    std::ifstream source_fin;

#ifndef __DEBUG__
    source_fin.open("Source.txt");
#else
    source_fin.open("../../PlotScript/Source.txt");
#endif

    if (!source_fin.is_open())
    {
        std::cout << "ERROR: Failed to open Source.txt" << std::endl;
        return 1;
    }

#ifdef __DEBUG__
    const char* argv[5] = { "1", "2", "3", "2", "1" };
#endif

    // Saving data from the console
    Component selected_field_1 = static_cast<Component>(std::atoi(argv[1]));
    Component selected_field_2 = static_cast<Component>(std::atoi(argv[2]));
    Component fld_tested = static_cast<Component>(std::atoi(argv[3]));
    bool version_flag = static_cast<bool>(std::atoi(argv[4]));

    Component flds_selected[2] = { selected_field_1, selected_field_2 };

    // Saving data from the txt file
    double number;
    while (source_fin >> number)
    {
        numbers.push_back(number);
    }
    int grid_sizes[2] = { numbers[0], numbers[1] };   // { Nx, Ny }
    double sizes_x[2] = { numbers[2], numbers[3] };   // { ax, bx }
    double sizes_y[2] = { numbers[4], numbers[5] };   // { ay, by }
    double step_sizes[2] = { (sizes_x[1] - sizes_x[0]) / grid_sizes[0],
                             (sizes_y[1] - sizes_y[0]) / grid_sizes[1] };   // { dx, dy }
    int iterations_num = static_cast<int>(numbers[6]);
    double time = numbers[7];
    double time_step = time / static_cast<double>(iterations_num);   // dt

    source_fin.close();

    // Initialization of the initializing function and the true solution function
    std::function<double(double, double[2])> initial_func =
        [](double x, double size[2]) {
        return sin(2.0 * M_PI * (x - size[0]) / (size[1] - size[0]));
    };
    std::function<double(double, double, double[2])> true_func =
        [](double x, double t, double size[2]) {
        return sin(2.0 * M_PI * (x - size[0] - FDTD_Const::C * t) / (size[1] - size[0]));
    };

    // Determination of the wave propagation axis
    char selected_axis;
    if (flds_selected[0] == Component::EY && flds_selected[1] == Component::BZ ||
        flds_selected[0] == Component::EZ && flds_selected[1] == Component::BY)
    {
        selected_axis = 'x';
    }
    else if (flds_selected[0] == Component::EX && flds_selected[1] == Component::BZ ||
        flds_selected[0] == Component::EZ && flds_selected[1] == Component::BX)
    {
        selected_axis = 'y';
    }
    else
    {
        std::cout << "ERROR" << std::endl;
        exit(1);
    }

    // Meaningful calculations
    FDTD test_1(grid_sizes, sizes_x, sizes_y, time_step);
    SelectedFields current_fields{ 
        flds_selected[0], 
        flds_selected[1], 
        fld_tested 
    };
    Parameters params{ 
        sizes_x[0], 
        sizes_x[1], 
        sizes_y[0], 
        sizes_y[1], 
        step_sizes[0], 
        step_sizes[1], 
        time, 
        iterations_num 
    };
    Functions funcs{
        initial_func,
        true_func
    };
    Test_FDTD test(test_1, current_fields, params, funcs, version_flag);
    std::cout << test.get_max_abs_error() << std::endl;

    // Writing the results to a file
    char* file_path;
#ifndef __DEBUG__
    file_path = "OutFile.csv";
#else
    file_path = "../../PlotScript/OutFile.csv";
#endif
    write_all(test_1, selected_axis, file_path);

    return 0;
}
