set(target "FDTD")

file(GLOB hdrs "../../include/FDTD/*.h")
file(GLOB srcs "../../src/FDTD/*.cpp")

add_library(${target} STATIC ${srcs} ${hdrs})
