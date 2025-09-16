#include <impl/graphics/allocate_descriptor_set.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto allocate_descriptor_set(VkDevice device,  //
                             VkDescriptorPool descriptorPool,  //
                             VkDescriptorSetLayout descriptorSetLayout  //
                             ) -> VkDescriptorSet {
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.pNext = nullptr;
    info.descriptorPool = descriptorPool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(device, &info, &set) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to allocate descriptor set."};
    }

    return set;
}

}  // namespace graphics
}  // namespace impl
