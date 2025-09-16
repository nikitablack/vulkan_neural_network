#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto get_compute_queue_family(VkPhysicalDevice device, uint32_t requiredQueueCount) -> uint32_t;

}  // namespace graphics
}  // namespace impl
