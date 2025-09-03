cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(eigen_nn LANGUAGES CXX)

add_executable(eigen_nn)

target_sources(eigen_nn
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Layer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/NeuralNetwork.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/NeuralNetwork.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(eigen_nn
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(eigen_nn
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(eigen_nn
    PRIVATE
        Eigen3::Eigen
        fmt::fmt
        libs::common
        libs::warnings_as_errors
)

option(VKNN_USE_AVX2 "Use AVX2 instructions" OFF)

if(VKNN_USE_AVX2)
    target_compile_options(eigen_nn
        PRIVATE
            -mavx2
    )
endif()
