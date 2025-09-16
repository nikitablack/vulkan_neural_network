#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace impl {
namespace graphics {

auto get_supported_physical_devices(VkInstance instance) -> std::vector<VkPhysicalDevice>;

}  // namespace graphics
}  // namespace impl
