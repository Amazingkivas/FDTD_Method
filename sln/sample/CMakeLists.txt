file(GLOB hdrs "*.h*" "../../include/*.h")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../samples/*.cpp")

add_executable(sample ${srcs} ${hdrs})

if(WIN32)
    # Для Windows
    set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../../PlotScript/src")
else()
    # Для Linux и других UNIX-подобных систем
    set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../../PlotScript/src/Release")
endif()
