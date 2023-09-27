file(GLOB hdrs "*.h*" "../include/*.h")
file(GLOB srcs "*.cpp" "../src/*.cpp")

add_compile_definitions(_PATH_=${"/FDTD_Method/PlotScript/"})

add_executable(FDTD ${srcs} ${hdrs})
