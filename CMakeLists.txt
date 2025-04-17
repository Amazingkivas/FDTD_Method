cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT "${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/build")
    message(FATAL_ERROR "The build must be performed in the 'build' folder. Please run the following commands:\n\n"
                        "  mkdir -p build && cd build && cmake ..\n\n"
                        "The 'build' folder has been created for you.")
endif()

project(FDTD)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DNDEBUG -xAVX -qopenmp -qopt-report=5 -qopt-report-phase=vec -fp-model precise")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OPENMP FOUND")
endif()

message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX Flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX Flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")

if(CMAKE_CXX_FLAGS MATCHES "-S")
    message(STATUS "Flag -S detected")
else()
    message(STATUS "Flag -S not found")
endif()

include_directories(3rdparty)

UNSET(Kokkos_DIR CACHE)
set(Kokkos_DIR 3rdparty/kokkos/build/)

if(NOT DEFINED Kokkos_ENABLE_CONTAINERS)
    set(Kokkos_ENABLE_CONTAINERS ON)
endif()

if(NOT DEFINED Kokkos_ENABLE_ALL_PACKAGES)
    set(Kokkos_ENABLE_ALL_PACKAGES ON)
endif()

set(KOKKOS_SIMD_LIB "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build/simd/src/libkokkossimd.a")
set(KOKKOS_CORE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build/core/src/libkokkoscore.a")
set(KOKKOS_CONTAINERS_LIB "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build/containers/src/libkokkoscontainers.a")

add_custom_command(
    OUTPUT ${KOKKOS_CONTAINERS_LIB}
    COMMAND ${CMAKE_COMMAND} -E echo "Building Kokkos Containers Library"
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build --target kokkoscontainers
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos
    COMMENT "Creating libkokkoscontainers.a"
)

add_custom_target(create_kokkoscontainers ALL DEPENDS ${KOKKOS_CONTAINERS_LIB})

add_custom_command(
    OUTPUT ${KOKKOS_SIMD_LIB}
    COMMAND ${CMAKE_COMMAND} -E echo "Building Kokkos SIMD Library"
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build --target kokkossimd
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos
    COMMENT "Creating libkokkossimd.a"
)

add_custom_target(create_kokkossimd ALL DEPENDS ${KOKKOS_SIMD_LIB})

add_custom_command(
    OUTPUT ${KOKKOS_CORE_LIB}
    COMMAND ${CMAKE_COMMAND} -E echo "Building Kokkos Core Library"
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build --target kokkoscore
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos
    COMMENT "Creating libkokkoscore.a"
)

add_custom_target(create_kokkoscore ALL DEPENDS ${KOKKOS_CORE_LIB})


find_package(Kokkos REQUIRED)

add_subdirectory(project)
add_subdirectory(test)
add_subdirectory(samples)
