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
## Download all submodules
  ```
  git submodule update --init --recursive
  ```

## Build the project with `CMake`
  
  ```
  cmake .
  cmake --build . --config RELEASE
  ```

## Run and visualize

![](https://github.com/Amazingkivas/FDTD_Method/blob/main/PlotScript/Animations/animation_Ex.gif)

### Get information about running the application and visualising the results
* **Linux (gcc)**:
  
  ```
  cd PlotScript
  python3 visualization.py --help
  ```
* **Windows (MSVC)**:
  
  ```
  cd PlotScript
  python visualization.py --help
  ```

