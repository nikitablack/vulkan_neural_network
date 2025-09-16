#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_update_pipeline(VkDevice device, VkPipelineLayout pipelineLayout) -> VkPipeline;

}  // namespace graphics
}  // namespace impl
