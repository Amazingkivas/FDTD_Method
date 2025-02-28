set(target test_FDTD_method)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")

include_directories(../include ../include/FDTD ../include/FDTD_kokkos ../src ../src/FDTD ../src/FDTD_kokkos)

file(GLOB hdrs "*../include/*.h" "../include/FDTD/*.h" "../include/FDTD_kokkos/*.h")
#file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../src/*.cpp" "../src/FDTD/*.cpp" "../src/FDTD_kokkos/*.cpp")

add_subdirectory(../3rdparty/googletest ../3rdparty/googletest/build EXCLUDE_FROM_ALL)

add_executable(${target} ${srcs} ${hdrs})
add_dependencies(${target} gtest)
target_link_directories(${target} PUBLIC ${CMAKE_BINARY_DIR}/googletest/lib)
target_link_libraries(${target} PUBLIC gtest gtest_main)

target_link_libraries(${target} PUBLIC Kokkos::kokkos)

add_compile_options(/wd4996)
