cmake_minimum_required(VERSION 3.15)
project(fair_spread_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

add_library(fair_spread_core STATIC
    src/diffusion.cpp
    src/metrics.cpp
)
target_include_directories(fair_spread_core PUBLIC include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fair_spread_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Standalone test executable
add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE fair_spread_core)

# pybind11 module if pybind11 is available
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(fair_spread_cpp src/bindings.cpp)
    target_link_libraries(fair_spread_cpp PRIVATE fair_spread_core)
endif()
