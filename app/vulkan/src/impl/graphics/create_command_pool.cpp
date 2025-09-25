#include <impl/graphics/create_command_pool.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_command_pool(VkDevice device, uint32_t queueFamilyIndex) -> VkCommandPool {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = queueFamilyIndex;

    VkCommandPool commandPool{VK_NULL_HANDLE};
    if (vkCreateCommandPool(device, &info, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create command pool."};
    }

    return commandPool;
}

}  // namespace graphics
}  // namespace impl
