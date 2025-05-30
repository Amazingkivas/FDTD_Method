[![Build|Test|Run](https://github.com/Amazingkivas/FDTD_Method/actions/workflows/main.yml/badge.svg)](https://github.com/Amazingkivas/FDTD_Method/actions/workflows/main.yml)
![Platforms](https://img.shields.io/badge/platform-linux-lightgrey.svg)

# FDTD Method

Finite-difference time-domain (FDTD) is a numerical analysis technique used for modeling computational electrodynamics. 

This repository contains a C++ project with the main implementation of the method.

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

# Visualization

![](https://github.com/Amazingkivas/FDTD_Method/blob/main/animations/animation_Ez.gif)

