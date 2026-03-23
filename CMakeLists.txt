cmake_minimum_required(VERSION 4.2.3)

project(graphics_simulator
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "A 2D graphics simulator using rasterisation and fake implmentation of CPU & GPU")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(X11 REQUIRED)

add_executable(simulator main.c)

target_include_directories(simulator PRIVATE ${X11_INCLUDE_DIRS})
target_link_libraries(simulator PRIVATE ${X11_LIBRARIES} m)