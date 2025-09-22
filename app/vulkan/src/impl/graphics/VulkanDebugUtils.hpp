#pragma once

#include <vulkan/vulkan.h>

#include <string>

namespace impl {
namespace graphics {

class VulkanDebugUtils {
public:
    VulkanDebugUtils() = default;
    auto initialize(VkDevice device) noexcept -> void;

    auto setName(VkBuffer object, std::string const& name) const noexcept -> void;
    auto setName(VkDescriptorPool object, std::string const& name) const noexcept -> void;
    auto setName(VkDescriptorSet object, std::string const& name) const noexcept -> void;
    auto setName(VkDescriptorSetLayout object, std::string const& name) const noexcept -> void;
    auto setName(VkFence object, std::string const& name) const noexcept -> void;
    auto setName(VkPipeline object, std::string const& name) const noexcept -> void;
    auto setName(VkPipelineLayout object, std::string const& name) const noexcept -> void;
    auto setName(VkQueue object, std::string const& name) const noexcept -> void;

private:
    auto setName(VkObjectType objectType, uint64_t objectHandle, std::string const& name) const noexcept -> void;

private:
    VkDevice m_device{VK_NULL_HANDLE};
};

}  // namespace graphics
}  // namespace impl
