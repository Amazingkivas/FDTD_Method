file(GLOB hdrs "*.h*" "../../include/*.h")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../samples/*.cpp")

add_executable(sample ${srcs} ${hdrs})

if(WIN32)
    set(output_dir "../../PlotScript/src")
else()
    set(output_dir "../../PlotScript/src/Release")
endif()

set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_dir})

if(NOT EXISTS ${output_dir})
    file(MAKE_DIRECTORY ${output_dir})
endif()
