import matplotlib.pyplot as plt
import csv
import subprocess


def get_plot(num, data, size_N, size_x, time):
    components = {1: "Ex", 2: "Ey", 3: "Ez", 4: "Bx", 5: "By", 6: "Bz"}
    x = 0.
    X = []
    V = []
    dx = (size_x[1] - size_x[0]) / size_N[0]
    while round(x, len(str(dx)) - 2) < size_x[1]:
        X.append(x)
        x += dx

    with open(data, 'r') as datafile:
        plotting = list(csv.reader(datafile, delimiter=';'))

        V.extend([float(value) for value in plotting[(num - 1) * (size_N[1] + 2)]])

    plt.plot(X, V)
    plt.xlabel('X')
    plt.ylabel(components[num])
    plt.title('Plot ' + components[num] + " (Max time: " + str(time) + ")")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    input_list = ["Ni", "Nj", "ax", "bx", "ay", "by", "dt", "max_t"]

    with open("Source.txt", "w") as file:
        for component in input_list:
            file.write(input(component + ": ") + "\n")

    cpp_executable = "src/sample.exe"
    try:
        subprocess.run(cpp_executable, check=True)
    except subprocess.CalledProcessError:
        print("Error when starting a C++ project")
    except FileNotFoundError:
        print("sample.exe not found")

    with open('Source.txt', 'r') as file:
        numbers = [float(line.strip()) for line in file]

    arr_N = (int(numbers[0]), int(numbers[1]))
    arr_x = (int(numbers[2]), int(numbers[3]))

    print("Select field:  \n \
          1 - Ex \n \
          2 - Ey \n \
          3 - Ez \n \
          4 - Bx \n \
          5 - By \n \
          6 - Bz")

    select = int(input("Number: "))
    if not (0 < select < 7):
        print("Invalid input")
        exit(1)

    file = 'OutFile.csv'
    get_plot(select, file, arr_N, arr_x, numbers[7])
