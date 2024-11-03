file(GLOB hdrs "*../../include/*.h" "../../include/FDTD_kokkos/*.h")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../src/FDTD_kokkos/*.cpp" "kokkos_sample.cpp")

add_executable(kokkos_sample ${srcs} ${hdrs})

if(WIN32)
    set(output_dir "../../PlotScript/src")
else()
    set(output_dir "../../PlotScript/src/Release")
endif()

target_link_libraries(kokkos_sample Kokkos::kokkos)

set_target_properties(kokkos_sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_dir})

if(NOT EXISTS ${output_dir})
    file(MAKE_DIRECTORY ${output_dir})
endif()
