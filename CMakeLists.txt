cmake_minimum_required(VERSION 3.1)
project(asst)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    a8_main.cpp
    basicImageManipulation.cpp
    basicImageManipulation.h
    blending.cpp
    blending.h
    filtering.cpp
    filtering.h
    homography.cpp
    homography.h
    Image.cpp
    Image.h
    ImageException.h
    lodepng.cpp
    lodepng.h
    matrix.h
    messageassert.h
    panorama.cpp
    panorama.h)

add_executable(asst ${SOURCE_FILES})