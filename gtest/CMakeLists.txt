set(target "gtest")

add_library(${target} STATIC gtest-all.cc)

if((${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR
    ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang") AND
    (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    set(pthread "-pthread")
endif()

add_compile_options(/wd4996)

target_link_libraries(${target} ${pthread})

set_target_properties(${target} PROPERTIES
    FOLDER "sln")
