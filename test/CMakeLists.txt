set(target FDTD_test)

#file(GLOB hdrs "*.h*" "../include/*.h" "../gtest/*.h")
file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../src/*.cpp")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} gtest)

set_target_properties(${target} PROPERTIES
    FOLDER "sln")

configure_file(${CMAKE_SOURCE_DIR}/test/FDTD_test.vcxproj ${CMAKE_BINARY_DIR}/sln/tests/FDTD_test.vcxproj COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/test/FDTD_test.vcxproj.filters ${CMAKE_BINARY_DIR}/sln/tests/FDTD_test.vcxproj.filters COPYONLY)