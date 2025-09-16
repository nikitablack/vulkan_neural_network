#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_descriptor_pool(VkDevice device) -> VkDescriptorPool;

}  // namespace graphics
}  // namespace impl
