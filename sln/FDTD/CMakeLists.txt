set(target "FDTD")

add_library(${target} STATIC ../../include/FDTD.h ../../include/Field.h ../../include/Current.h ../../src/FDTD.cpp ../../src/Field.cpp ../../src/Current.cpp)
