#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto allocate_descriptor_set(VkDevice device,  //
                             VkDescriptorPool descriptorPool,  //
                             VkDescriptorSetLayout descriptorSetLayout  //
                             ) -> VkDescriptorSet;

}  // namespace graphics
}  // namespace impl
