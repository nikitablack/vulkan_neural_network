cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(vulkan_nn LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_executable(vulkan_nn)

target_sources(vulkan_nn
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/features/Features.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/features/Features.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/barrier_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/barrier_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/get_push_constant_data.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/init_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/init_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/read_float_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/read_float_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/utils/update_descriptor_set.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/allocate_descriptor_set.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/allocate_descriptor_set.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/calculate_hidden_delta.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/calculate_hidden_delta.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/calculate_output_delta.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/calculate_output_delta.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/check_instance_version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/check_instance_version.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/check_physical_device_support.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/check_physical_device_support.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/check_required_instance_extensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/check_required_instance_extensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_allocator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_allocator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_batch_index_pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_batch_index_pipeline.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_command_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_command_pool.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_delta_pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_delta_pipeline.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_descriptor_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_descriptor_pool.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_descriptor_set_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_descriptor_set_layout.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_device.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_forward_pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_forward_pipeline.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_instance.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_instance.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_pipeline_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_pipeline_layout.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_shader_module.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_shader_module.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_update_pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/create_update_pipeline.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/DeviceBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/DeviceBuffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_command_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_command_buffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_compute_queue_family.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_compute_queue_family.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_compute_queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_compute_queue.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_physical_device_properties.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_physical_device_properties.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_supported_physical_devices.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/get_supported_physical_devices.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/GraphicsManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/GraphicsManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/HostVisibleBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/HostVisibleBuffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/RequiredApiVersion.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/RequiredDeviceExtensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/RequiredDeviceExtensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/RequiredInstanceExtensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/RequiredInstanceExtensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/set_input_and_target_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/set_input_and_target_data.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/submit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/submit.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/update_current_batch_index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/update_current_batch_index.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/VulkanDebugUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/VulkanDebugUtils.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/VulkanFunctions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/VulkanFunctions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/graphics/VulkanQueue.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/Layer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/NeuralNetwork.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/NeuralNetwork.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(vulkan_nn
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(vulkan_nn
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

option(VKNN_ENABLE_VULKAN_DEBUG_UTILS "Enable Vulkan debug utils" ON)

if(VKNN_ENABLE_VULKAN_DEBUG_UTILS)
    target_compile_definitions(vulkan_nn
        PRIVATE
            ENABLE_VULKAN_DEBUG_UTILS
    )
endif()

# shader compilation
if (NOT Vulkan_glslangValidator_FOUND)
    message(FATAL_ERROR "Failed to find glslangValidator.")
endif ()

cmrc_add_resource_library(shaders "")

function(addCompileShadersCommand)
    cmake_parse_arguments(addCompileShadersCommand "" "TARGET_NAME;OUTPUT_DIR" "SHADERS;DEPENDS" ${ARGN})

    if (NOT addCompileShadersCommand_TARGET_NAME)
        message(FATAL_ERROR "Provide unique target name.")
    endif ()

    if (NOT addCompileShadersCommand_SHADERS)
        message(FATAL_ERROR "At least one shader file name should be provided.")
    endif ()

    set(FILE_NAMES "")
    set(OUTPUT_PATHS "")

    foreach (shader ${addCompileShadersCommand_SHADERS})
        if(NOT EXISTS ${shader})
            message(FATAL_ERROR "Failed to find shader file ${shader}.")
        endif()

        file(MAKE_DIRECTORY ${addCompileShadersCommand_OUTPUT_DIR})

        get_filename_component(VAR ${shader} NAME)
        list(APPEND FILE_NAMES ${VAR})
        list(APPEND OUTPUT_PATHS ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv)

        add_custom_command(OUTPUT ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv
                COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -gVS --target-env vulkan1.3 -V ${shader} -o ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv
                DEPENDS ${shader} ${addCompileShadersCommand_DEPENDS}
                )

        cmrc_add_resources(shaders WHENCE ${addCompileShadersCommand_OUTPUT_DIR} ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv)
    endforeach ()

    add_custom_target(${addCompileShadersCommand_TARGET_NAME}
            DEPENDS ${OUTPUT_PATHS}
            COMMENT "Compiling ${FILE_NAMES}"
            )

    add_dependencies(vulkan_nn ${addCompileShadersCommand_TARGET_NAME})
endfunction()

addCompileShadersCommand(TARGET_NAME compile_shaders
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders
    SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/batch_index.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/forward.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/delta.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/update.comp
)

target_link_libraries(vulkan_nn
    PRIVATE
        fmt::fmt
        GPUOpen::VulkanMemoryAllocator
        libs::common
        libs::warnings_as_errors
        shaders
        Vulkan::Vulkan
)