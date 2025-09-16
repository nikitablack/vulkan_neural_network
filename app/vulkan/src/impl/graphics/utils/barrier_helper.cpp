#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>

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
                        ) noexcept -> void {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = size;

    vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        DeviceBuffer const& deviceBuffer,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer.getBuffer(),  //
                       deviceBuffer.getSize(),  //
                       srcStageMask,  //
                       srcAccessMask,  //
                       dstStageMask,  //
                       dstAccessMask);
}

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        HostVisibleBuffer const& vulkanBuffer,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       vulkanBuffer.getBuffer(),  //
                       vulkanBuffer.getSize(),  //
                       srcStageMask,  //
                       srcAccessMask,  //
                       dstStageMask,  //
                       dstAccessMask);
}

}  // namespace utils
}  // namespace graphics
}  // namespace impl
