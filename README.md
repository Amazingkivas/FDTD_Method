[![Build application](https://github.com/Amazingkivas/FDTD_Method/actions/workflows/main.yml/badge.svg)](https://github.com/Amazingkivas/FDTD_Method/actions/workflows/main.yml)

# FDTD Method

Finite-difference time-domain (FDTD) is a numerical analysis technique used for modeling computational electrodynamics. 

This repository contains a C++ project with the main implementation of the method. The following python script is used for testing and visualization:
* `PlotScript/visualization.py`
## Install packages

```
pip install pandas
pip install matplotlib
```

## Build the project with `CMake`
  
  ```
  cd sln
  cmake .
  cmake --build . --config RELEASE
  ```

## Run and visualise

![](https://github.com/Amazingkivas/FDTD_Method/blob/main/PlotScript/Animations/animation_Ex.gif)

### Go to the folder
```
cd PlotScript
```

### To run the method and save the data
* **Linux (gcc)**:
  
  ```
  python3 visualization.py --run_cpp --grid_size <grid size> --iters_num <iterations number> <component>
  ```
* **Windows (MSVC)**:
  
  ```
  python visualization.py --run_cpp --grid_size <grid size> --iters_num <iterations number> <component>
  ```
### To create an animation
* **Linux (gcc)**:
  
  ```
  python3 visualization.py --function animation <component for analysis>
  ```
* **Windows (MSVC)**:
  
  ```
  python visualization.py --function animation <component for analysis>
  ```
The result will be saved to a folder `PlotScript/animations`
### To create a heatmap
* **Linux (gcc)**:
  
  ```
  python3 visualization.py --function heatmap --iteration <iteration number> <component>
  ```
* **Windows (MSVC)**:
  
  ```
  python visualization.py --function heatmap --iteration <iteration number> <component>
  ```
The result will be saved to a folder `PlotScript/heatmap`
