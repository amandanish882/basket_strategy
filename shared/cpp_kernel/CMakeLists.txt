cmake_minimum_required(VERSION 3.15)
project(pricing_kernel LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pricing_kernel bindings/pybind_module.cpp)
target_include_directories(pricing_kernel PRIVATE include)
target_compile_features(pricing_kernel PRIVATE cxx_std_17)
