cmake_minimum_required(VERSION 3.10.2)
project(ai)

set(CMAKE_CXX_STANDARD 14)

add_executable(ai main.cpp ai.h neuron.h ai.cpp neuron.cpp)