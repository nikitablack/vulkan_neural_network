#include <cstring>
#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/get_command_buffer.hpp>
#include <impl/graphics/submit.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/read_float_helper.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {
namespace utils {

auto read_float_helper(GraphicsManager& graphicsManager,  //
                       DeviceBuffer const& buffer,  //
                       VkPipelineStageFlags srcStageMask,  //
                       VkAccessFlags srcAccessMask,  //
                       std::vector<float>& out  //
                       ) -> void {
    auto const commandBuffer{get_command_buffer_begin(graphicsManager.device, graphicsManager.commandPool)};

    set_buffer_barrier(commandBuffer,  //
                       buffer,  //
                       srcStageMask,  //
                       srcAccessMask,  //
                       VK_PIPELINE_STAGE_TRANSFER_BIT,  //
                       VK_ACCESS_TRANSFER_READ_BIT);

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = buffer.getSize();

    impl::graphics::HostVisibleBuffer dataHost{graphicsManager.allocator,  //
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                                               out.size() * sizeof(float),  //
                                               true};

    vkCmdCopyBuffer(commandBuffer, buffer.getBuffer(), dataHost.getBuffer(), 1, &region);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to end copy command buffer."};
    }

    submit(commandBuffer, graphicsManager.computeQueue.queue, VK_NULL_HANDLE);

    if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to wait staging queue."};
    }

    auto const* const data{static_cast<float const*>(dataHost.getMappedData())};
    std::memcpy(out.data(), data, out.size() * sizeof(float));

    dataHost.destroy();
}

}  // namespace utils
}  // namespace graphics
}  // namespace impl
