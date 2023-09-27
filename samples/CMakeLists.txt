file(GLOB hdrs "*.h*" "../include/*.h")
file(GLOB srcs "*.cpp" "../src/*.cpp")

set(RELATIVE_PATH_TO_FILE "FDTD_Method/src")
add_definitions(-DRELATIVE_PATH="${RELATIVE_PATH_TO_FILE}")

add_executable(FDTD ${srcs} ${hdrs})
