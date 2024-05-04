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

    fig, ax = plt.subplots()

    def update(frame):
        mut = 0
        ax.clear()
        data = pd.read_csv(file_names[frame], sep=';')
        data = data.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
        ax.imshow(data, cmap='viridis', vmin=-0.07, vmax=0.07, interpolation='none') # RdBu
        ax.set_title('Frame {}'.format(frame + 1))

    ani = FuncAnimation(fig, update, frames=len(file_names), interval=50)

    plt.show()
    ani.save(f'animations/animation_{component}.gif', writer='imagemagick')


def get_heatmap(component, iteration):
    data = pd.read_csv(f'OutFiles_{components_num[component]}/{iteration}.csv', sep=';')

    data = data.applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()


def execute_cpp(grid_size, iters_num, single_iteration_flag=True):
    cpp_executable = 'src/Release/sample.exe'
    args = [cpp_executable, str(grid_size), str(iters_num), str(int(single_iteration_flag))]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print('Error when starting a C++ project')
    except FileNotFoundError:
        print('sample.exe not found')


if __name__ == '__main__':
    #execute_cpp(75, 410, False)
    get_animation("Ex")
