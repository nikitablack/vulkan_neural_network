#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <impl/graphics/CommandManager.hpp>
#include <impl/graphics/VulkanDebugUtils.hpp>
#include <impl/graphics/VulkanQueue.hpp>
#include <vector>

// forward declarations
namespace impl {
namespace graphics {

class DeviceBuffer;

}  // namespace graphics
}  // namespace impl

namespace impl {
namespace graphics {

class GraphicsManager {
public:
    auto init() -> void;
    auto changePhysicalDevice() -> void;
    auto clear() noexcept -> void;
    auto flush() const noexcept -> void;

public:
    VkInstance instance{VK_NULL_HANDLE};
    std::vector<VkPhysicalDevice> supportedPhysicalDevices{};

    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    VkDevice device{VK_NULL_HANDLE};
    VulkanDebugUtils debugUtils{};
    VulkanQueue computeQueue{};
    VmaAllocator allocator{VK_NULL_HANDLE};
    CommandManager commandManager{};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> storageDescriptorSets{};
    VkPipeline deltaPipeline{VK_NULL_HANDLE};
    VkPipeline forwardPipeline{VK_NULL_HANDLE};
    VkPipeline updatePipeline{VK_NULL_HANDLE};
};

}  // namespace graphics
}  // namespace impl
