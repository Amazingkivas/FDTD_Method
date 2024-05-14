cmake_minimum_required(VERSION 2.8)

project(FDTD)

include_directories(../include ../gtest ../test)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# BUILD
add_subdirectory(FDTD)
add_subdirectory(FDTD_test)
add_subdirectory(test_FDTD_method)
add_subdirectory(gtest)
add_subdirectory(sample)
