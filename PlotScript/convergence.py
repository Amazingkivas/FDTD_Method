import matplotlib.pyplot as plt
import csv
import subprocess

components = {1: "Ex", 2: "Ey", 3: "Ez", 4: "Bx", 5: "By", 6: "Bz"}
nums_com = {"Ex": 0, "Ey": 1, "Ez": 2, "Bx": 3, "By": 4, "Bz": 5}
input_list = ["Ni", "Nj", "ax", "bx", "ay", "by", "dt", "t"]


def execute_cpp(field_1, field_2, field_to_plot, shift_flag):
    num_field_1 = nums_com[field_1]
    num_field_2 = nums_com[field_2]
    num_field_to_plot = nums_com[field_to_plot]

    print("\n" + field_to_plot + ":\n")

    cpp_executable = "src/Release/sample.exe"
    args = [cpp_executable, str(num_field_1), str(num_field_2), str(num_field_to_plot), shift_flag]
    try:
        completed_process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
        output = completed_process.stdout.decode("utf-8").strip()
        print(output)
        return output
    except subprocess.CalledProcessError:
        print("Error when starting a C++ project")
    except FileNotFoundError:
        print("sample.exe not found")


def update_sources():
    print("Update parameters? \n \
                          1 - Yes \n \
                          2 - No")
    select_update = int(input("Number: ")) * (-1) + 2
    if not (select_update == 0 or select_update == 1):
        print("Invalid input")
        exit(1)
    elif (select_update == 1):
        with open("Source.txt", "w") as file:
            for component in input_list:
                file.write(input(component + ": ") + "\n")


def save_source_into_reserve():
    with open('Source.txt', 'r') as file1, open('Reserve.txt', 'w') as file2:
        numbers = [float(line.strip()) for line in file1]
        for num in numbers:
            file2.write(str(num) + "\n")

    return numbers


def reload_source():
    with open('Source.txt', 'w') as file2, open('Reserve.txt', 'r') as file1:
        numbers = [float(line.strip()) for line in file1]
        for num in numbers:
            file2.write(str(num) + "\n")


def analyze_convergence(numbers, mult_2, iterations, shifts):
    convergences = []
    nums = []
    flag = str(int(shifts))
    for n in range(0, iterations):
        with open("Source.txt", "w") as file:
            mult_1 = 2

            tmp_n_0 = numbers[0] * (mult_1 ** n)
            file.write(str(tmp_n_0) + "\n")

            tmp_n_1 = numbers[1] * (mult_1 ** n)
            file.write(str(tmp_n_1) + "\n")

            file.write(str(numbers[2]) + "\n")
            file.write(str(numbers[3]) + "\n")
            file.write(str(numbers[4]) + "\n")
            file.write(str(numbers[5]) + "\n")

            tmp_n_6 = numbers[6] * (mult_2 ** n)
            file.write(str(tmp_n_6) + "\n")
            file.write(str(numbers[7]) + "\n")

        convergences.append(float(execute_cpp("Ex", "Bz", "Ex", flag)))

    convers = []
    for n in range(0, iterations - 1):
        convers.append(convergences[n] / convergences[n + 1])
        nums.append(n)
    return nums, convers


if __name__ == '__main__':
    update_sources()
    loaded_numbers = save_source_into_reserve()
    nums, convergence = analyze_convergence(loaded_numbers, 4, 5, False)
    reload_source()

    print(convergence)
    plt.plot(nums, convergence)
    plt.xlabel('n')
    plt.ylabel('E/mult')
    plt.title("Plot")
    plt.grid(True)
    plt.show()
