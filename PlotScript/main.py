import matplotlib.pyplot as plt
import csv
import subprocess


def get_plot(num, data, size_N, size_x, time):
    components = {1: "Ex", 2: "Ey", 3: "Ez", 4: "Bx", 5: "By", 6: "Bz"}
    x = 0.
    X = []
    V = []
    dx = (float(size_x[1]) - float(size_x[0])) / float(size_N[0])
    cnt = 0
    while cnt < size_N[0]:
        X.append(x)
        x += dx
        cnt += 1

    with open(data, 'r') as datafile:
        plotting = list(csv.reader(datafile, delimiter=';'))

        V.extend([float(value) for value in plotting[(num - 1) * (size_N[1] + 2)]])

    plt.plot(X, V)
    plt.xlabel('X')
    plt.ylabel(components[num])
    plt.title('Plot ' + components[num] + " (Time: " + str(time) + ")")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    input_list = ["Ni", "Nj", "ax", "bx", "ay", "by", "dt", "t"]

    print("Update parameters?  \n \
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

    print("Select the active fields:  \n \
                  1 - Ex \n \
                  2 - Ey \n \
                  3 - Ez \n \
                  4 - Bx \n \
                  5 - By \n \
                  6 - Bz")
    field_1 = -1 + int(input("Number: "))
    field_2 = -1 + int(input("Number: "))
    if not (0 <= field_1 < 6 and 0 <= field_2 < 6):
        print("Invalid input")
        exit(1)

    print("Select field for plotting:  \n \
              1 - Ex \n \
              2 - Ey \n \
              3 - Ez \n \
              4 - Bx \n \
              5 - By \n \
              6 - Bz")
    select_to_plot = -1 + int(input("Number: "))
    if not (0 <= select_to_plot < 6 and (select_to_plot in (field_1, field_2))):
        print("Invalid input")
        exit(1)

    with open('Source.txt', 'r') as file:
        numbers = [float(line.strip()) for line in file]

    if (field_1 == 0 and field_2 == 5 or field_1 == 2 and field_2 == 3):
        arr_N = (int(numbers[1]), int(numbers[0]))
        arr_x = (float(numbers[4]), float(numbers[5]))
    elif (field_1 == 1 and field_2 == 5 or field_1 == 2 and field_2 == 4):
        arr_N = (int(numbers[0]), int(numbers[1]))
        arr_x = (float(numbers[2]), float(numbers[3]))
    else:
        print("Invalid selected fields")
        exit(1)

    cpp_executable = "src/sample.exe"
    args = [cpp_executable, str(field_1), str(field_2), str(select_to_plot)]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print("Error when starting a C++ project")
    except FileNotFoundError:
        print("sample.exe not found")

    file = 'OutFile.csv'
    get_plot(select_to_plot + 1, file, arr_N, arr_x, numbers[7])
