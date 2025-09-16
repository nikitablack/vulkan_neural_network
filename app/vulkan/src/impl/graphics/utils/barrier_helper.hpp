#pragma once

#include <vulkan/vulkan.h>

// forward declaration
namespace graphics {

class DeviceBuffer;
class HostVisibleBuffer;

}  // namespace graphics

namespace impl {
namespace graphics {
namespace utils {

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        VkBuffer buffer,  //
                        VkDeviceSize size,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void;

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        DeviceBuffer const& deviceBuffer,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void;

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        HostVisibleBuffer const& vulkanBuffer,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void;

}  // namespace utils
}  // namespace graphics
}  // namespace impl
