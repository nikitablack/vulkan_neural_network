#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_device(VkPhysicalDevice physicalDevice,  //
                   uint32_t graphicsQueueFamily,  //
                   uint32_t queueCount  //
                   ) -> VkDevice;

}  // namespace graphics
}  // namespace impl
