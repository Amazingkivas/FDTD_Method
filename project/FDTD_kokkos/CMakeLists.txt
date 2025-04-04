set(target "FDTD_kokkos")

file(GLOB hdrs "../../src/FDTD_kokkos/*.h")
file(GLOB srcs "../../src/*.cpp")

add_library(${target} STATIC ${srcs} ${hdrs})

target_link_libraries(${target} Kokkos::kokkos)