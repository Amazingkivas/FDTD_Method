file(GLOB hdrs "*.h*" "../include/*.h")
file(GLOB srcs "*.cpp" "../src/*.cpp")

add_executable(dijkstra ${srcs} ${hdrs})
