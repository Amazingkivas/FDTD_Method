set(target "FDTD")

add_library(${target} STATIC ../../include/FDTD.h ../../include/Field.h ../../src/FDTD.cpp ../../src/Field.cpp)
