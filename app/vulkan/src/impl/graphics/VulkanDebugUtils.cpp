#include <impl/graphics/VulkanDebugUtils.hpp>
#include <impl/graphics/VulkanFunctions.hpp>

namespace impl {
namespace graphics {

auto VulkanDebugUtils::initialize(VkDevice device) noexcept -> void {
    m_device = device;
}

auto VulkanDebugUtils::setName(VkBuffer object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkDescriptorSet object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_DESCRIPTOR_SET, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkDescriptorSetLayout object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkFence object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_FENCE, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkPipeline object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkQueue object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkPipelineLayout object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName(VkDescriptorPool object, std::string const& name) const noexcept -> void {
    setName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, reinterpret_cast<uint64_t>(object), name);
}

auto VulkanDebugUtils::setName([[maybe_unused]] VkObjectType objectType, [[maybe_unused]] uint64_t objectHandle,
                               [[maybe_unused]] std::string const& name) const noexcept -> void {
#ifdef ENABLE_VULKAN_DEBUG_UTILS
    if (!m_device) {
        return;
    }

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.pNext = nullptr;
    info.objectType = objectType;
    info.objectHandle = objectHandle;
    info.pObjectName = name.c_str();

    VulkanFunctions::vkSetDebugUtilsObjectNameEXT(m_device, &info);
#endif
}

}  // namespace graphics
}  // namespace impl
