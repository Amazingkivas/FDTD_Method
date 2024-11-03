[![Build|Test|Run](https://github.com/Amazingkivas/FDTD_Method/actions/workflows/main.yml/badge.svg)](https://github.com/Amazingkivas/FDTD_Method/actions/workflows/main.yml)
![Platforms](https://img.shields.io/badge/platform-linux-lightgrey.svg)

# FDTD Method

Finite-difference time-domain (FDTD) is a numerical analysis technique used for modeling computational electrodynamics. 

This repository contains a C++ project with the main implementation of the method. The following python script is used for testing and visualization:
* `PlotScript/visualization.py`

## 0. Download all submodules
  ```
  git submodule update --init --recursive
  ```
## 1. Build Kokkos
  ```
  cd 3rdparty/kokkos
  mkdir build
  cd build
  cmake .. -DKokkos_ENABLE_OPENMP=ON
  cd ../../..
  ```
## 2. Build the project
  ```
  cmake .
  cmake --build . --config RELEASE
  ```

## Run and visualize

![](https://github.com/Amazingkivas/FDTD_Method/blob/main/PlotScript/Animations/animation_Ez.gif)

### 1. Setting up a virtual environment
  ```
  cd PlotScript
  python3 -m venv venv
  source venv/bin/activate
  ```
### 2. Install packages
  ```
  pip install pandas
  pip install matplotlib
  ```
### 3. Get information about running the application and visualising the results
  ```
  python3 visualization.py --help
  ```
