#include <array>
#include <impl/graphics/create_descriptor_set_layout.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_descriptor_set_layout(VkDevice device,  //
                                  VkDescriptorType descriptorType  //
                                  ) -> VkDescriptorSetLayout {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    for (size_t i{0}; i < bindings.size(); i++) {
        bindings[i].binding = static_cast<uint32_t>(i);
        bindings[i].descriptorType = descriptorType;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.pBindings = bindings.data();

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create descriptor set layout."};
    }

    return descriptorSetLayout;
}

}  // namespace graphics
}  // namespace impl
