cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(naive LANGUAGES CXX)

add_executable(naive)

target_sources(naive
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Float.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Layer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/load_images.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/load_images.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/load_labels.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/load_labels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/NeuralNetwork.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/NeuralNetwork.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Neuron.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Neuron.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/ScopedTimer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/ScopedTimer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/swap_endian.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/swap_endian.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Timer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Timer.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(naive
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(naive
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(naive
    PRIVATE
        cmrc_dataset
        common::warnings_as_errors
        fmt::fmt
)

option(USE_DOUBLE "Use double precision" OFF)

if(USE_DOUBLE)
    target_compile_definitions(naive
        PRIVATE
            -DUSE_DOUBLE
    )
endif()

# target_compile_options(naive
#     PRIVATE
#         -save-temps=obj
#         -fverbose-asm
#         -masm=intel
#         -mavx2
#         -mfma
# )
