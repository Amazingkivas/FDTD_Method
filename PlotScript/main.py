import matplotlib.pyplot as plt
import csv
import subprocess


components = {1: "Ex", 2: "Ey", 3: "Ez", 4: "Bx", 5: "By", 6: "Bz"}
nums_com = {"Ex": 0, "Ey": 1, "Ez": 2, "Bx": 3, "By": 4, "Bz": 5}
shift_flag = "1"

def get_plot(num, data, size_n, size_x, time):
    x = 0.
    X = []
    V = []
    dx = (float(size_x[1]) - float(size_x[0])) / float(size_n[0])
    cnt = 0
    while cnt < size_n[0]:
        X.append(x)
        x += dx
        cnt += 1

    with open(data, 'r') as datafile:
        plotting = list(csv.reader(datafile, delimiter=';'))

        V.extend([float(value) for value in plotting[(num - 1) * (size_n[1] + 2)]])

    plt.plot(X, V)
    plt.xlabel('X')
    plt.ylabel(components[num])
    plt.title(f"Plot {components[num]} (Time: {str(time)})")
    plt.grid(True)
    plt.savefig(f"Plots/plt_{components[num]}")
    plt.show()


def execute_cpp(field_1, field_2, field_to_plot, source_nums):
    num_field_1 = nums_com[field_1]
    num_field_2 = nums_com[field_2]
    num_field_to_plot = nums_com[field_to_plot]

    print("\n" + field_to_plot + ":\n")

    cpp_executable = "src/sample.exe"
    args = [cpp_executable, str(num_field_1), str(num_field_2), str(num_field_to_plot), shift_flag]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print("Error when starting a C++ project")
    except FileNotFoundError:
        print("sample.exe not found")

    if (num_field_1 == 0 and num_field_2 == 5 or num_field_1 == 2 and num_field_2 == 3):
        arr_n = (int(source_nums[1]), int(source_nums[0]))
        arr_x = (float(source_nums[4]), float(source_nums[5]))

    elif (num_field_1 == 1 and num_field_2 == 5 or num_field_1 == 2 and num_field_2 == 4):
        arr_n = (int(source_nums[0]), int(source_nums[1]))
        arr_x = (float(source_nums[2]), float(source_nums[3]))
    else:
        print("Invalid selected fields")
        exit(1)

    file = 'OutFile.csv'
    get_plot(num_field_to_plot + 1, file, arr_n, arr_x, source_nums[7])


if __name__ == '__main__':
    input_list = ["Ni", "Nj", "ax", "bx", "ay", "by", "dt", "t"]

    print("Update parameters? \n \
                      1 - Yes \n \
                      2 - No")
    select_update = int(input("Number: ")) * (-1) + 2
    if not (select_update == 0 or select_update == 1):
        print("Invalid input")
        exit(1)

    if (select_update):
        with open("Source.txt", "w") as file:
            for component in input_list:
                file.write(input(component + ": ") + "\n")
    else:
        with open("Source.txt", "r+") as file:
            lines = file.readlines()
            lines[-1] = input("t : ") + "\n"
            file.seek(0)
            file.writelines(lines)
            file.truncate()

    with open('Source.txt', 'r') as file:
        numbers = [float(line.strip()) for line in file]

    execute_cpp("Ex", "Bz", "Ex", numbers)
    execute_cpp("Ey", "Bz", "Ey", numbers)
    execute_cpp("Ez", "By", "Ez", numbers)
    execute_cpp("Ez", "Bx", "Bx", numbers)
    execute_cpp("Ez", "By", "By", numbers)
    execute_cpp("Ex", "Bz", "Bz", numbers)
