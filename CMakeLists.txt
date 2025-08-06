cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(template LANGUAGES CXX)

include(FetchContent)

option(BUILD_TESTS "Whether or not to build the tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(third_party/catch SYSTEM)
endif()

add_subdirectory(third_party/fmt SYSTEM)

add_subdirectory(libs/warnings)

add_subdirectory(app)
