#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto check_physical_device_support(VkPhysicalDevice device) -> void;

}  // namespace graphics
}  // namespace impl
