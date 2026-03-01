include_directories(../include ../include/FDTD ../include/FDTD_kokkos ../src ../src/FDTD ../src/FDTD_kokkos)

add_subdirectory(sample)
add_subdirectory(kokkos_sample)

add_executable(vectorization_compare vectorization_compare.cpp)
target_link_libraries(vectorization_compare Kokkos::kokkos)
set_target_properties(vectorization_compare PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
