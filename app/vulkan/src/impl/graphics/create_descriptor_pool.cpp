#include <array>
#include <impl/graphics/create_descriptor_pool.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_descriptor_pool(VkDevice device) -> VkDescriptorPool {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.maxSets = 100;
    info.poolSizeCount = 1;
    info.pPoolSizes = &poolSize;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create descriptor pool."};
    }

    return descriptorPool;
}

}  // namespace graphics
}  // namespace impl
