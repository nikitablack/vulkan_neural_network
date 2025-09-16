#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vector>

namespace impl {
namespace graphics {

class HostVisibleBuffer {
public:
    HostVisibleBuffer() = default;
    HostVisibleBuffer(VmaAllocator allocator,  //
                      VkBufferUsageFlags usageFlags,  //
                      size_t size,  //
                      bool readback = false);

public:
    auto init(VmaAllocator allocator,  //
              VkBufferUsageFlags usageFlags,  //
              size_t size,  //
              bool readback = false  //
              ) -> void;
    auto copyData(std::vector<uint8_t> const& data, size_t offset = 0) noexcept -> void;
    auto copyData(uint8_t const* data, size_t size, size_t offset = 0) noexcept -> void;
    auto destroy() noexcept -> void;
    auto getSize() const noexcept -> size_t;
    auto getBuffer() const noexcept -> VkBuffer;
    auto getMappedData() const noexcept -> void const*;

private:
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
    void* m_mappedData{nullptr};
};

}  // namespace graphics
}  // namespace impl
