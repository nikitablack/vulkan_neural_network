cmake_minimum_required(VERSION 3.25)

set(VMA_VERSION v3.3.0)
message(STATUS "Downloading vma ${VMA_VERSION}")

FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG ${VMA_VERSION}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src
)

FetchContent_MakeAvailable(vma)
