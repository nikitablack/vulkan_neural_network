#pragma once

#include <impl/graphics/HostVisibleBuffer.hpp>
#include <unordered_map>

// forward declarations
namespace graphics {

class DeviceBuffer;
class GraphicsManager;
struct VulkanQueue;

}  // namespace graphics

namespace impl {
namespace graphics {
namespace utils {

[[nodiscard]] auto init_buffer(VkCommandBuffer commandBuffer,  //
                               DeviceBuffer& deviceBuffer,  //
                               std::vector<uint8_t> const& data,  //
                               VkPipelineStageFlags dstStageMask,  //
                               VkAccessFlags dstAccessMask  //
                               ) -> HostVisibleBuffer;

[[nodiscard]] auto init_buffer(VkCommandBuffer commandBuffer,  //
                               DeviceBuffer& deviceBuffer,  //
                               uint8_t const* data,  //
                               size_t size,  //
                               VkPipelineStageFlags dstStageMask,  //
                               VkAccessFlags dstAccessMask  //
                               ) -> HostVisibleBuffer;

auto init_buffer(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer& deviceBuffer,  //
                 HostVisibleBuffer const& stagingBuffer,  //
                 VkPipelineStageFlags dstStageMask,  //
                 VkAccessFlags dstAccessMask  //
                 ) noexcept -> void;

auto init_buffer_sync(GraphicsManager& graphicsManager,  //
                      DeviceBuffer& deviceBuffer,  //
                      std::vector<uint8_t> const& data,  //
                      VkPipelineStageFlags dstStageMask,  //
                      VkAccessFlags dstAccessMask,  //
                      VulkanQueue const& queue  //
                      ) -> void;

auto init_buffer_sync(GraphicsManager& graphicsManager,  //
                      DeviceBuffer& deviceBuffer,  //
                      uint8_t const* data,  //
                      size_t size,  //
                      VkPipelineStageFlags dstStageMask,  //
                      VkAccessFlags dstAccessMask,  //
                      VulkanQueue const& queue  //
                      ) -> void;

auto init_buffer_sync(
    GraphicsManager& graphicsManager,  //
    DeviceBuffer& deviceBuffer,  //
    HostVisibleBuffer&& stagingBuffer,  // pass ownership to the function. The buffer will be destroyed.
    VkPipelineStageFlags dstStageMask,  //
    VkAccessFlags dstAccessMask,  //
    VulkanQueue const& queue  //
    ) -> void;

auto submit_init_data_sync(std::unordered_map<VkCommandBuffer, std::vector<HostVisibleBuffer>>&& initData,  //
                           VulkanQueue const& queue  //
                           ) -> void;

}  // namespace utils
}  // namespace graphics
}  // namespace impl
