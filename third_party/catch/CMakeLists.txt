cmake_minimum_required(VERSION 3.25)

set(CATCH_VERSION v3.9.0)
message(STATUS "Downloading Catch2 ${CATCH_VERSION}")

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG ${CATCH_VERSION}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src
)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
