import matplotlib.pyplot as plt
import csv


def get_plot(num, data, size_N, size_x):
    components = {1: "Ex", 2: "Ey", 3: "Ez", 4: "Bx", 5: "By", 6: "Bz"}
    x = 0.
    X = []
    V = []
    dx = (size_x[1] - size_x[0]) / size_N[0]
    while round(x, len(str(dx)) - 2) < size_x[1]:
        x += dx
        X.append(round(x, len(str(dx)) - 2))

    with open(data, 'r') as datafile:
        plotting = list(csv.reader(datafile, delimiter=';'))

        V.extend([float(value) for value in plotting[(num - 1) * (size_N[1] + 2) + 1]])

    plt.plot(X, V)
    plt.xlabel('X')
    plt.ylabel('V')
    plt.title('Plot ' + components[num])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    with open('/FDTD_Method/src/Source.txt', 'r') as file:
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
    get_plot(select, file, arr_N, arr_x)
