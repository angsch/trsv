cmake_minimum_required(VERSION 3.16)

project(trsv)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_definitions(FORTRAN_STRLEN_END)

set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(test test.cpp generator.cpp trsv.cpp)
target_link_libraries(test ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} m)

add_executable(bench bench.cpp generator.cpp trsv.cpp)
target_link_libraries(bench ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} m)
