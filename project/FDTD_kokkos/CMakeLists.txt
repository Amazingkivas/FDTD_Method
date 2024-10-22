set(target "FDTD_kokkos")

file(GLOB hdrs "../../include/FDTD_kokkos/*.h")
file(GLOB srcs "../../src/FDTD_kokkos/*.cpp")

add_library(${target} STATIC ${srcs} ${hdrs})

target_link_libraries(${target} Kokkos::kokkos)