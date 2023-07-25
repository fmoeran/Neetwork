cmake_minimum_required(VERSION 3.25)
project(AI)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB SOURCE_FILES src/*.hpp src/*.cpp src/*.tpp)

#add_compile_definitions(SIMD)
add_compile_options(-mavx)

add_executable(AI ${SOURCE_FILES})
