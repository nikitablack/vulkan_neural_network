#define VMA_IMPLEMENTATION
#include <fmt/core.h>

#include <array>
#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/VulkanFunctions.hpp>
#include <impl/graphics/allocate_descriptor_set.hpp>
#include <impl/graphics/check_instance_version.hpp>
#include <impl/graphics/check_required_instance_extensions.hpp>
#include <impl/graphics/create_allocator.hpp>
#include <impl/graphics/create_delta_pipeline.hpp>
#include <impl/graphics/create_descriptor_pool.hpp>
#include <impl/graphics/create_descriptor_set_layout.hpp>
#include <impl/graphics/create_device.hpp>
#include <impl/graphics/create_forward_pipeline.hpp>
#include <impl/graphics/create_instance.hpp>
#include <impl/graphics/create_pipeline_layout.hpp>
#include <impl/graphics/create_update_pipeline.hpp>
#include <impl/graphics/get_compute_queue.hpp>
#include <impl/graphics/get_compute_queue_family.hpp>
#include <impl/graphics/get_physical_device_properties.hpp>
#include <impl/graphics/get_supported_physical_devices.hpp>

namespace impl {
namespace graphics {

auto GraphicsManager::init() -> void {
    check_instance_version();
    check_required_instance_extensions();

    instance = create_instance();

    VulkanFunctions::initialize(instance);

    supportedPhysicalDevices = get_supported_physical_devices(instance);

    changePhysicalDevice();
}

auto GraphicsManager::changePhysicalDevice() -> void {
    // clear all previously created objects, if any
    clear();

    physicalDevice = VK_NULL_HANDLE;

    // TODO: add select-by-index

    // if no devices were saved, take the first discrete device
    if (physicalDevice == VK_NULL_HANDLE) {
        for (auto const d : supportedPhysicalDevices) {
            if (get_physical_device_properties(d).deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = d;
                break;
            }
        }
    }

    // if no discrete devices were found, take the first device
    if (physicalDevice == VK_NULL_HANDLE) {
        physicalDevice = supportedPhysicalDevices[0];
    }

    physicalDeviceProperties = get_physical_device_properties(physicalDevice);

    fmt::println("Selected physical device: {}", physicalDeviceProperties.deviceName);

    uint32_t constexpr REQUIRED_COMPUTE_QUEUE_COUNT{1};
    uint32_t computeQueueFamily{get_compute_queue_family(physicalDevice, REQUIRED_COMPUTE_QUEUE_COUNT)};

    device = create_device(physicalDevice, computeQueueFamily, REQUIRED_COMPUTE_QUEUE_COUNT);

    debugUtils.initialize(device);

    uint32_t constexpr COMPUTE_QUEUE_INDEX{0};
    static_assert(COMPUTE_QUEUE_INDEX < REQUIRED_COMPUTE_QUEUE_COUNT);
    computeQueue.queueFamily = computeQueueFamily;
    computeQueue.queue = get_compute_queue(device, computeQueueFamily, COMPUTE_QUEUE_INDEX);

    debugUtils.setName(computeQueue.queue, fmt::format("Compute queue {}.", COMPUTE_QUEUE_INDEX));

    allocator = create_allocator(instance, physicalDevice, device);

    commandManager.init(device, computeQueueFamily);

    descriptorSetLayout = create_descriptor_set_layout(device, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    debugUtils.setName(descriptorSetLayout, "Storage descriptor set layout.");

    pipelineLayout = create_pipeline_layout(device, descriptorSetLayout);
    debugUtils.setName(pipelineLayout, "Pipeline layout.");

    descriptorPool = create_descriptor_pool(device);
    debugUtils.setName(descriptorPool, "Descriptor pool.");

    deltaPipeline = create_delta_pipeline(device, pipelineLayout);
    debugUtils.setName(deltaPipeline, "Delta pipeline.");

    forwardPipeline = create_forward_pipeline(device, pipelineLayout);
    debugUtils.setName(forwardPipeline, "Forward pipeline.");

    updatePipeline = create_update_pipeline(device, pipelineLayout);
    debugUtils.setName(updatePipeline, "Update pipeline.");
}

auto GraphicsManager::clear() noexcept -> void {
    if (!device) {
        return;
    }

    flush();

    vkDestroyPipeline(device, updatePipeline, nullptr);
    updatePipeline = VK_NULL_HANDLE;

    vkDestroyPipeline(device, forwardPipeline, nullptr);
    forwardPipeline = VK_NULL_HANDLE;

    vkDestroyPipeline(device, deltaPipeline, nullptr);
    deltaPipeline = VK_NULL_HANDLE;

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    descriptorPool = VK_NULL_HANDLE;

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    descriptorSetLayout = VK_NULL_HANDLE;

    commandManager.clear();

    vmaDestroyAllocator(allocator);
    allocator = VK_NULL_HANDLE;

    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;

    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
}

auto GraphicsManager::flush() const noexcept -> void {
    auto const r{vkDeviceWaitIdle(device)};
    if (r != VK_SUCCESS) {
        fmt::println("Failed to synchronize in impl::graphics::GraphicsManager::flush(): {}", static_cast<int32_t>(r));
    }
}

}  // namespace graphics
}  // namespace impl
