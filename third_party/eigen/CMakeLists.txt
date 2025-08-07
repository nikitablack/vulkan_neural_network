cmake_minimum_required(VERSION 3.25)

set(EIGEN_VERSION 3.4)
message(STATUS "Downloading eigen ${EIGEN_VERSION}")

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://github.com/PX4/eigen.git
    GIT_TAG ${EIGEN_VERSION}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src
)

FetchContent_MakeAvailable(eigen)
