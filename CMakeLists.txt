cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(vknn LANGUAGES CXX)

include(FetchContent)

option(BUILD_TESTS "Whether or not to build the tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(third_party/catch SYSTEM)
endif()

add_subdirectory(third_party/cmakerc SYSTEM)
add_subdirectory(third_party/eigen SYSTEM)
add_subdirectory(third_party/fmt SYSTEM)

add_subdirectory(libs/warnings)

add_subdirectory(app)

cmrc_add_resource_library(cmrc_dataset "")

cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/train-images.idx3-ubyte)
cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/train-labels.idx1-ubyte)
cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/t10k-images.idx3-ubyte)
cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/t10k-labels.idx1-ubyte)
