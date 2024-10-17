set(target FDTD_test)

file(GLOB hdrs "*.h*" "../../include/*.h")
#file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../../src/*.cpp")

add_library(${target} ${srcs} ${hdrs})

add_compile_options(/wd4996)
