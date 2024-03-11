import matplotlib.pyplot as plt
import csv
import subprocess


components = {1: 'Ex', 2: 'Ey', 3: 'Ez', 4: 'Bx', 5: 'By', 6: 'Bz'}
nums_com = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Bx': 3, 'By': 4, 'Bz': 5}
shift_flag = "1"


def get_plot(field, data_path, size_axis, data_block_height, axis_boundaries, time):
    field_num = nums_com[field] + 1
    a = 0.
    A = []
    V = []
    dA = (float(axis_boundaries[1]) - float(axis_boundaries[0])) / float(size_axis)

    for n in range(size_axis):
        A.append(a)
        a += dA
    with open(data_path, 'r') as datafile:
        plotting = list(csv.reader(datafile, delimiter=';'))
        V.extend([float(value) for value in plotting[(field_num - 1) * (data_block_height + 2)]])

    plt.plot(A, V)
    plt.xlabel('Coordinate')
    plt.ylabel(components[field_num])
    plt.title(f'Plot {components[field_num]} (Time: {str(time)})')
    plt.grid(True)
    plt.savefig(f'Plots/plt_{components[field_num]}')
    plt.show()


def select_parameters(field_E, field_B, source_nums):
    num_field_E = nums_com[field_E]
    num_field_B = nums_com[field_B]

    if num_field_E == 0 and num_field_B == 5 or num_field_E == 2 and num_field_B == 3:
        boundaries = (float(source_nums[3]), float(source_nums[4]))
        data_size = source_nums[1] * source_nums[2]
        axis_size = source_nums[0]
    elif num_field_E == 1 and num_field_B == 5 or num_field_E == 2 and num_field_B == 4:
        boundaries = (float(source_nums[5]), float(source_nums[6]))
        data_size = source_nums[0] * source_nums[2]
        axis_size = source_nums[1]
    elif num_field_E == 0 and num_field_B == 4 or num_field_E == 1 and num_field_B == 3:
        boundaries = (float(source_nums[7]), float(source_nums[8]))
        data_size = source_nums[0] * source_nums[1]
        axis_size = source_nums[2]
    else:
        print('Invalid selected fields')
        exit(1)

    return int(axis_size), boundaries, int(data_size)


def execute_cpp(field_E, field_B, field_to_plot):
    num_field_E = nums_com[field_E]
    num_field_B = nums_com[field_B]
    num_field_to_plot = nums_com[field_to_plot]

    print("\n" + field_to_plot + ":\n")

    cpp_executable = 'src/Release/sample.exe'
    args = [cpp_executable, str(num_field_E), str(num_field_B), str(num_field_to_plot), shift_flag]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print('Error when starting a C++ project')
    except FileNotFoundError:
        print('sample.exe not found')


if __name__ == '__main__':
    input_list = ['Ni', 'Nj', 'Nk' 'ax', 'bx', 'ay', 'by', 'az', 'bz', 'Nt', 't']

    print('Update parameters? \n \
                      1 - Yes \n \
                      2 - No')
    select_update = int(input('Number: ')) * (-1) + 2
    if not (select_update == 0 or select_update == 1):
        print('Invalid input')
        exit(1)

    if (select_update):
        with open('Source.txt', 'w') as file:
            for component in input_list:
                file.write(input(component + ': ') + '\n')

    with open('Source.txt', 'r') as file:
        numbers = [float(line.strip()) for line in file]

    field_E = 'Ez'
    field_B = 'Bx'
    field_to_plot = 'Ez'

    execute_cpp(field_E, field_B, field_to_plot)
    size_axis, axis_boundaries, block_size = select_parameters(field_E, field_B, numbers)
    file = 'OutFile.csv'
    get_plot(field_to_plot, file, size_axis, block_size, axis_boundaries, numbers[10])
