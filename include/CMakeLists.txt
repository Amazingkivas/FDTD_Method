set(target "FDTD")

add_library(${target} STATIC FDTD.h ../src/FDTD.cpp)

set_target_properties(${target} PROPERTIES
    FOLDER "sln")
