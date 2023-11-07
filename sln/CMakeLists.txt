cmake_minimum_required(VERSION 2.8)

project(FDTD)

include_directories(../include ../gtest)


# BUILD
add_subdirectory(FDTD)
add_subdirectory(FDTD_test)
add_subdirectory(gtest)
add_subdirectory(sample)
