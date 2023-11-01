file(GLOB hdrs "*.h*" "../include/*.h")
file(GLOB srcs "*.cpp" "../src/*.cpp")

add_executable(sample ${srcs} ${hdrs})
set_target_properties(sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/PlotScript/src")

set_target_properties(sample PROPERTIES
    FOLDER "sln")

configure_file(${CMAKE_SOURCE_DIR}/samples/sample.vcxproj ${CMAKE_BINARY_DIR}/sln/samples/sample.vcxproj COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/samples/sample.vcxproj.filters ${CMAKE_BINARY_DIR}/sln/samples/sample.vcxproj.filters COPYONLY)