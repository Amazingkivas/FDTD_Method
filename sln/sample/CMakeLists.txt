file(GLOB hdrs "*.h*" "../../include/*.h")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../samples/*.cpp")

add_executable(sample ${srcs} ${hdrs})
set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../../PlotScript/src")

