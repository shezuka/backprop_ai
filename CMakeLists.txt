cmake_minimum_required(VERSION 3.15)
project(ai)

set(CMAKE_CXX_STANDARD 14)

add_executable(ai main.cpp ai.hpp neuron.hpp)