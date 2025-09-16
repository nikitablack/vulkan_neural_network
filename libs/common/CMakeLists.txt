cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(common LANGUAGES CXX)

add_library(common STATIC)

add_library(libs::common
    ALIAS
        common
)

target_sources(common
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/Float.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/LCG.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/load_images.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/load_labels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/ScopedTimer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/swap_endian.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common/Timer.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/load_images.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/load_labels.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/ScopedTimer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/swap_endian.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/Timer.cpp
)

target_include_directories(common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(common
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(common
    PRIVATE
        cmrc_dataset
        fmt::fmt
        libs::warnings_as_errors
)

option(VKNN_USE_DOUBLE "Use double precision" OFF)

if(VKNN_USE_DOUBLE)
    target_compile_definitions(common
        PUBLIC
            -DUSE_DOUBLE
    )
endif()
