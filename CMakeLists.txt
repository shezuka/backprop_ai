cmake_minimum_required(VERSION 3.15)
project(ai)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(ai
        main.cpp
        src/ai.cpp
        src/neuron.cpp
        src/bias.cpp
        src/layer.cpp
        src/connection.cpp
)
target_link_libraries(ai jsoncpp)