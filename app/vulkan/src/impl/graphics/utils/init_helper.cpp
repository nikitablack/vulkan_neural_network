#include <impl/graphics/CommandManager.hpp>
#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/VulkanQueue.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/init_helper.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {
namespace utils {

auto init_buffer(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer& deviceBuffer,  //
                 std::vector<uint8_t> const& data,  //
                 VkPipelineStageFlags dstStageMask,  //
                 VkAccessFlags dstAccessMask  //
                 ) -> HostVisibleBuffer {
    return init_buffer(commandBuffer,  //
                       deviceBuffer,  //
                       data.data(),  //
                       data.size(),  //
                       dstStageMask,  //
                       dstAccessMask);
}

auto init_buffer(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer& deviceBuffer,  //
                 uint8_t const* data,  //
                 size_t size,  //
                 VkPipelineStageFlags dstStageMask,  //
                 VkAccessFlags dstAccessMask  //
                 ) -> HostVisibleBuffer {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,  //
                       VK_ACCESS_NONE,  //
                       VK_PIPELINE_STAGE_TRANSFER_BIT,  //
                       VK_ACCESS_TRANSFER_WRITE_BIT);

    auto const stagingBuffer{deviceBuffer.copyData(commandBuffer, data, size)};

    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_TRANSFER_BIT,  //
                       VK_ACCESS_TRANSFER_WRITE_BIT,  //
                       dstStageMask,  //
                       dstAccessMask);

    return stagingBuffer;
}

auto init_buffer(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer& deviceBuffer,  //
                 HostVisibleBuffer const& stagingBuffer,  //
                 VkPipelineStageFlags dstStageMask,  //
                 VkAccessFlags dstAccessMask  //
                 ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,  //
                       VK_ACCESS_NONE,  //
                       VK_PIPELINE_STAGE_TRANSFER_BIT,  //
                       VK_ACCESS_TRANSFER_WRITE_BIT);

    deviceBuffer.copyData(commandBuffer, stagingBuffer);

    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_TRANSFER_BIT,  //
                       VK_ACCESS_TRANSFER_WRITE_BIT,  //
                       dstStageMask,  //
                       dstAccessMask);
}

auto init_buffer_sync(CommandManager& commandManager,  //
                      DeviceBuffer& deviceBuffer,  //
                      std::vector<uint8_t> const& data,  //
                      VkPipelineStageFlags dstStageMask,  //
                      VkAccessFlags dstAccessMask,  //
                      VulkanQueue const& queue  //
                      ) -> void {
    return init_buffer_sync(commandManager,  //
                            deviceBuffer,  //
                            data.data(),  //
                            data.size(),  //
                            dstStageMask,  //
                            dstAccessMask,  //
                            queue);
}

auto init_buffer_sync(CommandManager& commandManager,  //
                      DeviceBuffer& deviceBuffer,  //
                      uint8_t const* data,  //
                      size_t size,  //
                      VkPipelineStageFlags dstStageMask,  //
                      VkAccessFlags dstAccessMask,  //
                      VulkanQueue const& queue  //
                      ) -> void {
    auto const commandBuffer{commandManager.getCommandBufferBegin()};

    auto const stagingBuffer{init_buffer(commandBuffer,  //
                                         deviceBuffer,  //
                                         data,  //
                                         size,  //
                                         dstStageMask,  //
                                         dstAccessMask)};

    std::unordered_map<VkCommandBuffer, std::vector<HostVisibleBuffer>> cbToStagingBuffers{};
    cbToStagingBuffers[commandBuffer].push_back(stagingBuffer);

    return submit_init_data_sync(std::move(cbToStagingBuffers), queue);
}

auto init_buffer_sync(CommandManager& commandManager,  //
                      DeviceBuffer& deviceBuffer,  //
                      HostVisibleBuffer&& stagingBuffer,  //
                      VkPipelineStageFlags dstStageMask,  //
                      VkAccessFlags dstAccessMask,  //
                      VulkanQueue const& queue  //
                      ) -> void {
    auto const commandBuffer{commandManager.getCommandBufferBegin()};

    init_buffer(commandBuffer,  //
                deviceBuffer,  //
                stagingBuffer,  //
                dstStageMask,  //
                dstAccessMask);

    std::unordered_map<VkCommandBuffer, std::vector<HostVisibleBuffer>> cbToStagingBuffers{};
    cbToStagingBuffers[commandBuffer].push_back(stagingBuffer);

    return submit_init_data_sync(std::move(cbToStagingBuffers), queue);
}

auto submit_init_data_sync(std::unordered_map<VkCommandBuffer, std::vector<HostVisibleBuffer>>&& initData,  //
                           VulkanQueue const& queue  //
                           ) -> void {
    std::vector<VkCommandBuffer> submitInfos{};
    submitInfos.reserve(initData.size());

    for (auto const& data : initData) {
        const auto cb{data.first};

        if (vkEndCommandBuffer(cb) != VK_SUCCESS) {
            throw std::runtime_error{"Failed to end copy command buffer."};
        }

        submitInfos.push_back(cb);
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = static_cast<uint32_t>(submitInfos.size());
    submitInfo.pCommandBuffers = submitInfos.data();
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    if (vkQueueSubmit(queue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to submit staging command buffer."};
    }

    if (vkQueueWaitIdle(queue.queue) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to wait staging queue."};
    }

    for (auto& data : initData) {
        for (auto& stagingBuffer : data.second) {
            stagingBuffer.destroy();
        }
    }
}

}  // namespace utils
}  // namespace graphics
}  // namespace impl
