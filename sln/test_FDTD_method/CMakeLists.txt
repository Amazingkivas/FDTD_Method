set(target test_FDTD_method)

file(GLOB hdrs "*.h*" "../../include/*.h")
#file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../../src/*.cpp" "../../test/*.cpp")

add_subdirectory(../../3rdparty/googletest ../../3rdparty/googletest/build EXCLUDE_FROM_ALL)

add_executable(${target} ${srcs} ${hdrs})
add_dependencies(${target} gtest)
target_link_directories(${target} PUBLIC ${CMAKE_BINARY_DIR}/googletest/lib)
target_link_libraries(${target} PUBLIC gtest gtest_main)

add_compile_options(/wd4996)
