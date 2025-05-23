cmake_minimum_required(VERSION 3.10)
project(SkewCorrection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/lib)

set(SOURCES
    src/main.cpp
)

add_executable(skew_correction ${SOURCES})

target_link_libraries(skew_correction ${PNG_LIBRARIES})

if(UNIX)
    target_link_libraries(skew_correction m)
endif()