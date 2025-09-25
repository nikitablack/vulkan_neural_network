#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto get_command_buffer(VkDevice device,  //
                        VkCommandPool commandPool  //
                        ) -> VkCommandBuffer;

auto get_command_buffer_begin(VkDevice device,  //
                              VkCommandPool commandPool,  //
                              VkCommandBufferUsageFlags flags = 0  //
                              ) -> VkCommandBuffer;

}  // namespace graphics
}  // namespace impl
