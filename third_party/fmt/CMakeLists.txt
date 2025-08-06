cmake_minimum_required(VERSION 3.25)

set(FMT_VERSION 11.2.0)
message(STATUS "Downloading fmt ${FMT_VERSION}")

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG ${FMT_VERSION}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src
)

FetchContent_MakeAvailable(fmt)
