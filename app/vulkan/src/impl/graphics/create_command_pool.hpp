#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_command_pool(VkDevice device, uint32_t queueFamilyIndex) -> VkCommandPool;

}  // namespace graphics
}  // namespace impl
