cmake_minimum_required(VERSION 2.8)

project(FDTD)

include_directories(include gtest)

# BUILD
add_subdirectory(include)
add_subdirectory(samples)
add_subdirectory(test)
add_subdirectory(gtest)

set_target_properties(FDTD PROPERTIES
    FOLDER "sln")

configure_file(${CMAKE_SOURCE_DIR}/FDTD.sln ${CMAKE_BINARY_DIR}/sln/FDTD.sln COPYONLY)