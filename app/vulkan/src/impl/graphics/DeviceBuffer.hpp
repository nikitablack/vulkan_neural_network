#pragma once

#include <impl/graphics/HostVisibleBuffer.hpp>

namespace impl {
namespace graphics {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

public:
    auto init(VmaAllocator allocator,  //
              VkBufferUsageFlags usageFlags,  //
              size_t size  //
              ) -> void;

    /**
     * A temporary host-visible buffer is created and Vulkan copy command is issued. All the synchronization should
     * happen on the caller side.
     *
     * @return The function returns the temporary staging buffer. The ownership of this temporary staging buffer is
     * transferred to the caller. The caller should destroy it when the copy operation is completed.
     */
    auto copyData(VkCommandBuffer commandBuffer,  //
                  std::vector<uint8_t> const& data,  //
                  size_t offset = 0  //
                  ) -> HostVisibleBuffer;

    auto copyData(VkCommandBuffer commandBuffer,  //
                  uint8_t const* data,  //
                  size_t size,  //
                  size_t offset = 0  //
                  ) -> HostVisibleBuffer;

    auto destroy() noexcept -> void;
    auto getSize() const noexcept -> size_t;
    auto getBuffer() const noexcept -> VkBuffer;

private:
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
};

}  // namespace graphics
}  // namespace impl
