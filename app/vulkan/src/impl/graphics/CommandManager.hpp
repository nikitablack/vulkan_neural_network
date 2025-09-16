#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace impl {
namespace graphics {

class CommandManager {
public:
    CommandManager() = default;
    auto init(VkDevice device, uint32_t queueFamily) -> void;
    auto getCommandBuffer() -> VkCommandBuffer;
    auto getCommandBufferBegin() -> VkCommandBuffer;
    auto tick() -> void;
    auto clear() noexcept -> void;

private:
    VkDevice m_device{VK_NULL_HANDLE};
    VkCommandPool m_commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> m_availableCommandBuffers{};
    std::vector<VkCommandBuffer> m_busyCommandBuffers{};
};

}  // namespace graphics
}  // namespace impl
