#include <impl/graphics/get_command_buffer.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto get_command_buffer(VkDevice device,  //
                        VkCommandPool commandPool  //
                        ) -> VkCommandBuffer {
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.pNext = nullptr;
    info.commandPool = commandPool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;

    VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
    if (vkAllocateCommandBuffers(device, &info, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to allocate command buffer."};
    }

    return commandBuffer;
}

auto get_command_buffer_begin(VkDevice device,  //
                              VkCommandPool commandPool,  //
                              VkCommandBufferUsageFlags flags  //
                              ) -> VkCommandBuffer {
    auto const commandBuffer{get_command_buffer(device, commandPool)};

    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = flags;
    info.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &info) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to begin command buffer."};
    }

    return commandBuffer;
}

}  // namespace graphics
}  // namespace impl
