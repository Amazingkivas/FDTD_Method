file(GLOB hdrs "*../../include/*.h" "../../include/FDTD/*.h")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../src/FDTD/*.cpp" "sample.cpp")

add_executable(sample ${srcs} ${hdrs})

set(output_dir "${CMAKE_SOURCE_DIR}/bin")

set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_dir})

if(NOT EXISTS ${output_dir})
    file(MAKE_DIRECTORY ${output_dir})
endif()
