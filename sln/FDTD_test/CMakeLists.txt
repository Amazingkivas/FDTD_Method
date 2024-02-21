set(target FDTD_test)

file(GLOB hdrs "*.h*" "../../include/*.h")
#file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../../src/*.cpp")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} gtest)

add_compile_options(/wd4996)
