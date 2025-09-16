#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto get_physical_device_properties(VkPhysicalDevice device) noexcept -> VkPhysicalDeviceProperties;

}  // namespace graphics
}  // namespace impl
