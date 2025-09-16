#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_pipeline_layout(VkDevice device,  //
                            VkDescriptorSetLayout descriptorSetLayout  //
                            ) -> VkPipelineLayout;

}  // namespace graphics
}  // namespace impl
