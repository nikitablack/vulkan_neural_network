cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(app LANGUAGES CXX)

add_executable(app)

target_sources(app
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
        cmrc_dataset
        common::warnings_as_errors
        fmt::fmt
)

# target_compile_definitions(app
#     PRIVATE
#         -DUSE_DOUBLE
# )
