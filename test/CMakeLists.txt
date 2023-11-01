set(target FDTD_test)

#file(GLOB hdrs "*.h*" "../include/*.h" "../gtest/*.h")
file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp" "../src/*.cpp")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} gtest)

set_target_properties(${target} PROPERTIES
    FOLDER "sln")
