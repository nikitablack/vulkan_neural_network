#include <impl/graphics/create_forward_pipeline.hpp>
#include <impl/graphics/create_shader_module.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_update_pipeline(VkDevice device, VkPipelineLayout pipelineLayout) -> VkPipeline {
    auto const computeShaderModule{create_shader_module(device, "update.comp.spv")};

    VkPipelineShaderStageCreateInfo shaderStageInfo{};

    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.pNext = nullptr;
    shaderStageInfo.flags = 0;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = computeShaderModule;
    shaderStageInfo.pName = "main";
    shaderStageInfo.pSpecializationInfo = nullptr;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.flags = 0;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create update pipeline."};
    }

    vkDestroyShaderModule(device, computeShaderModule, nullptr);

    return pipeline;
}

}  // namespace graphics
}  // namespace impl
