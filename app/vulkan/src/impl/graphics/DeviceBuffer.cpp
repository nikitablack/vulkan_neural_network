#include <cstring>
#include <impl/graphics/DeviceBuffer.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto DeviceBuffer::init(VmaAllocator allocator,  //
                        VkBufferUsageFlags usageFlags,  //
                        size_t size  //
                        ) -> void {
    m_allocator = allocator;
    m_size = size;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = nullptr;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags = 0;
    allocationCreateInfo.preferredFlags = 0;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = nullptr;

    VmaAllocationInfo allocationInfo;
    if (vmaCreateBuffer(m_allocator, &bufferCreateInfo, &allocationCreateInfo, &m_buffer, &m_allocation,
                        &allocationInfo) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create device buffer."};
    }
}

auto DeviceBuffer::destroy() noexcept -> void {
    if (!m_allocator) {
        return;
    }

    vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);

    m_allocator = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_size = 0;
}

auto DeviceBuffer::copyData(VkCommandBuffer commandBuffer,  //
                            std::vector<uint8_t> const& data,  //
                            size_t offset  //
                            ) -> HostVisibleBuffer {
    return copyData(commandBuffer, data.data(), data.size(), offset);
}

auto DeviceBuffer::copyData(VkCommandBuffer commandBuffer,  //
                            uint8_t const* data,  //
                            size_t size,  //
                            size_t offset  //
                            ) -> HostVisibleBuffer {
    if (!m_allocator) {
        throw std::runtime_error{"Attempt to copy data to a not initialized instance of DeviceBuffer."};
    }

    HostVisibleBuffer stagingBuffer{};

    stagingBuffer.init(m_allocator, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, size);
    stagingBuffer.copyData(data, size, 0);

    copyData(commandBuffer, stagingBuffer, offset);

    return stagingBuffer;
}

auto DeviceBuffer::copyData(VkCommandBuffer commandBuffer,  //
                            HostVisibleBuffer const& stagingBuffer,  //
                            size_t offset  //
                            ) noexcept -> void {
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = offset;
    region.size = stagingBuffer.getSize();

    vkCmdCopyBuffer(commandBuffer, stagingBuffer.getBuffer(), m_buffer, 1, &region);
}

auto DeviceBuffer::getSize() const noexcept -> size_t {
    return m_size;
}

auto DeviceBuffer::getBuffer() const noexcept -> VkBuffer {
    return m_buffer;
}

}  // namespace graphics
}  // namespace impl
