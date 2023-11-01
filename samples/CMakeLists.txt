file(GLOB hdrs "*.h*" "../include/*.h")
file(GLOB srcs "*.cpp" "../src/*.cpp")

add_executable(sample ${srcs} ${hdrs})
set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/PlotScript/src")

