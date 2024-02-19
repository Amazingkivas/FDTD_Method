#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>

#include "test_FDTD.h"
#include "Writer.h"

char get_axis(Component field_E, Component field_B)
{
    char selected_axis;
    if (field_E == Component::EY && field_B == Component::BZ ||
        field_E == Component::EZ && field_B == Component::BY)
    {
        selected_axis = 'x';
    }
    else if (field_E == Component::EX && field_B == Component::BZ ||
             field_E == Component::EZ && field_B == Component::BX)
    {
        selected_axis = 'y';
    }
    else if (field_E == Component::EX && field_B == Component::BY ||
             field_E == Component::EY && field_B == Component::BX)
    {
        selected_axis = 'z';
    }
    else
    {
        std::cout << "ERROR" << std::endl;
        exit(1);
    }
    return selected_axis;
}

int main(int argc, char* argv[])
{
    std::ifstream source_fin;
    char* outfile_path;

    std::vector<char*> arguments(argv, argv + argc);
    if (argc == 1) 
    {
        source_fin.open("../../PlotScript/Source.txt");
        outfile_path = "../../PlotScript/OutFile.csv";

        const size_t size_tmp = 5;
        char* tmp[size_tmp]{ "1", "2", "3", "3", "1" };
        
        arguments.clear();
        arguments.reserve(size_tmp);
        std::copy(std::begin(tmp), std::end(tmp), std::back_inserter(arguments));
    }
    else
    {
        source_fin.open("Source.txt");
        outfile_path = "OutFile.csv";
    }
    if (!source_fin.is_open())
    {
        std::cout << "ERROR: Failed to open Source.txt" << std::endl;
        return 1;
    }


    // Saving data from the console
    Component selected_field_1 = static_cast<Component>(std::atoi(arguments[1]));
    Component selected_field_2 = static_cast<Component>(std::atoi(arguments[2]));
    Component fld_tested = static_cast<Component>(std::atoi(arguments[3]));
    bool version_flag = static_cast<bool>(std::atoi(arguments[4]));

    Component flds_selected[2] = { selected_field_1, selected_field_2 };


    // Saving data from the txt file
    std::vector<double> numbers;
    double number;
    while (source_fin >> number)
    {
        numbers.push_back(number);
    }
    int grid_sizes[3] = { numbers[0], numbers[1], numbers[2] };   // { Nx, Ny , Nz}
    double sizes_x[2] = { numbers[3], numbers[4] };               // { ax, bx }
    double sizes_y[2] = { numbers[5], numbers[6] };               // { ay, by }
    double sizes_z[2] = { numbers[7], numbers[8] };               // { az, bz }

    double step_sizes[3] = { (sizes_x[1] - sizes_x[0]) / grid_sizes[0],
                             (sizes_y[1] - sizes_y[0]) / grid_sizes[1], 
                             (sizes_z[1] - sizes_z[0]) / grid_sizes[2] };   // { dx, dy, dz }

    int iterations_num = static_cast<int>(numbers[9]);
    double time = numbers[10];
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
    char axis = get_axis(flds_selected[0], flds_selected[1]);


    // Meaningful calculations
    FDTD test_1(grid_sizes, sizes_x, sizes_y, sizes_z, time_step);
    SelectedFields current_fields
    { 
        flds_selected[0], 
        flds_selected[1], 
        fld_tested 
    };
    Parameters params
    { 
        sizes_x[0], 
        sizes_x[1], 
        sizes_y[0], 
        sizes_y[1], 
        sizes_z[0],
        sizes_z[1],
        step_sizes[0], 
        step_sizes[1], 
        step_sizes[2],
        time, 
        iterations_num 
    };
    Functions funcs
    {
        &initial_func,
        &true_func
    };
    Test_FDTD test(current_fields, params, funcs);
    test.run_test(test_1);
    std::cout << test.get_max_abs_error() << std::endl;


    // Writing the results to a file
    write_all(test_1, axis, outfile_path);

    return 0;
}
