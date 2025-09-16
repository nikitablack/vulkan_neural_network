#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_allocator(VkInstance instance,  //
                      VkPhysicalDevice physicalDevice,  //
                      VkDevice device  //
                      ) -> VmaAllocator;

}  // namespace graphics
}  // namespace impl
