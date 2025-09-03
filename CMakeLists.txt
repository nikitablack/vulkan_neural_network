cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(vknn LANGUAGES CXX)

include(FetchContent)

option(VKNN_BUILD_NAIVE "Build naive implementation" ON)
option(VKNN_BUILD_EIGEN "Build Eigen implementation" ON)

add_subdirectory(third_party/cmakerc SYSTEM)
add_subdirectory(third_party/fmt SYSTEM)

add_subdirectory(libs/common)
add_subdirectory(libs/warnings)

if(VKNN_BUILD_NAIVE)
    add_subdirectory(app/naive)
endif()

if(VKNN_BUILD_EIGEN)
    add_subdirectory(third_party/eigen SYSTEM)

    add_subdirectory(app/eigen)
endif()

cmrc_add_resource_library(cmrc_dataset "")

cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/train-images.idx3-ubyte)
cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/train-labels.idx1-ubyte)
cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/t10k-images.idx3-ubyte)
cmrc_add_resources(cmrc_dataset WHENCE ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_SOURCE_DIR}/resources/t10k-labels.idx1-ubyte)
