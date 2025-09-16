#include <cstring>
#include <impl/graphics/HostVisibleBuffer.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

HostVisibleBuffer::HostVisibleBuffer(VmaAllocator allocator,  //
                                     VkBufferUsageFlags usageFlags,  //
                                     size_t size,  //
                                     bool readback) {
    init(allocator, usageFlags, size, readback);
}

auto HostVisibleBuffer::init(VmaAllocator allocator,  //
                             VkBufferUsageFlags usageFlags,  //
                             size_t size,  //
                             bool readback  //
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

    VmaAllocationCreateFlags allocationFlags{};
    if (readback) {
        allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags = allocationFlags;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags = 0;
    allocationCreateInfo.preferredFlags = 0;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = nullptr;

    VmaAllocationInfo allocationInfo;
    if (vmaCreateBuffer(m_allocator, &bufferCreateInfo, &allocationCreateInfo, &m_buffer, &m_allocation,
                        &allocationInfo) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create host-visible buffer."};
    }

    // If the allocation hasn't been mapped using vmaMapMemory() and hasn't been created with
    // #VMA_ALLOCATION_CREATE_MAPPED_BIT flag, this value is null.
    m_mappedData = allocationInfo.pMappedData;

    if (m_mappedData == nullptr) {
        throw std::runtime_error{"Failed to create host-visible buffer."};
    }
}

auto HostVisibleBuffer::copyData(std::vector<uint8_t> const& data, size_t offset) noexcept -> void {
    copyData(data.data(), data.size(), offset);
}

auto HostVisibleBuffer::copyData(uint8_t const* data, size_t size, size_t offset) noexcept -> void {
    std::memcpy(reinterpret_cast<uint8_t*>(m_mappedData) + offset, data, size);
}

auto HostVisibleBuffer::destroy() noexcept -> void {
    if (!m_allocator) {
        return;
    }

    vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);

    m_allocator = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_size = 0;
    m_mappedData = nullptr;
}

auto HostVisibleBuffer::getSize() const noexcept -> size_t {
    return m_size;
}
auto HostVisibleBuffer::getBuffer() const noexcept -> VkBuffer {
    return m_buffer;
}

auto HostVisibleBuffer::getMappedData() const noexcept -> void const* {
    return m_mappedData;
}

}  // namespace graphics
}  // namespace impl
