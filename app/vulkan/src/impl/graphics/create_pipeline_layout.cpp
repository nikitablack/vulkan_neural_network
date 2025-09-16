#include <array>
#include <impl/graphics/create_pipeline_layout.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_pipeline_layout(VkDevice device,  //
                            VkDescriptorSetLayout descriptorSetLayout  //
                            ) -> VkPipelineLayout {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 128;

    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.setLayoutCount = 1;
    info.pSetLayouts = &descriptorSetLayout;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &pushConstantRange;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &info, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create pipeline layout."};
    }

    return pipelineLayout;
}

}  // namespace graphics
}  // namespace impl
