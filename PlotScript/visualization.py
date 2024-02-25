import glob
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

components_num = {"Ex": '1', "Ey": '2', "Ez": '3', "Bx": '4', "By": '5', "Bz": '6'}


def get_animation(component):
    folder_path = f'OutFiles_{components_num[component]}'
    file_names_base = sorted(glob.glob(folder_path + '/*.csv'))
    file_names = sorted(file_names_base, key=lambda x: int(x.split('\\')[-1].split('.')[0]))

    data_frames = [pd.read_csv(file) for file in file_names]

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        data = pd.read_csv(file_names[frame], sep=';')
        data = data.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
        ax.imshow(data, cmap='viridis', interpolation='nearest')
        ax.set_title('Frame {}'.format(frame + 1))

    ani = FuncAnimation(fig, update, frames=len(file_names), interval=20)
    plt.show()
    ani.save(f'animations/animation_{component}.gif', writer='imagemagick')


def execute_cpp(grid_size, iters_num):
    cpp_executable = 'src/Release/sample.exe'
    args = [cpp_executable, str(grid_size), str(iters_num)]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print('Error when starting a C++ project')
    except FileNotFoundError:
        print('sample.exe not found')


if __name__ == '__main__':
    execute_cpp(100, 12)
    get_animation("By")
