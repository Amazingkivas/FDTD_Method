set(target test_FDTD_method)

file(GLOB hdrs "*.h*" "../../include/*.h" "../../3rdparty/googletest/googletest/include/gtest/gtest.h")
#file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../test/*.cpp")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} gtest)

add_compile_options(/wd4996)
