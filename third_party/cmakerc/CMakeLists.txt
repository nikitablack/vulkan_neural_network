cmake_minimum_required(VERSION 3.25)

set(CMAKERC_VERSION 2.0.1)
message(STATUS "Downloading cmakerc ${CMAKERC_VERSION}")

FetchContent_Declare(
    cmakerc
    GIT_REPOSITORY https://github.com/vector-of-bool/cmrc.git
    GIT_TAG ${CMAKERC_VERSION}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src
)

FetchContent_MakeAvailable(cmakerc)

include(${cmakerc_SOURCE_DIR}/CMakeRC.cmake)
