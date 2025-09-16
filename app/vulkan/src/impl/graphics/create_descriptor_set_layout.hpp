#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_descriptor_set_layout(VkDevice device,  //
                                  VkDescriptorType descriptorType  //
                                  ) -> VkDescriptorSetLayout;

}  // namespace graphics
}  // namespace impl
