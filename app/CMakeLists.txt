cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(app LANGUAGES CXX)

add_executable(app)

target_sources(app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(app
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(app
    PRIVATE
        common::warnings_as_errors
        fmt::fmt
)
