
#include <impl/graphics/RequiredApiVersion.hpp>
#include <impl/graphics/create_allocator.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_allocator(VkInstance instance,  //
                      VkPhysicalDevice physicalDevice,  //
                      VkDevice device  //
                      ) -> VmaAllocator {
    VmaAllocatorCreateInfo info{};
    info.flags = 0;
    info.physicalDevice = physicalDevice;
    info.device = device;
    info.preferredLargeHeapBlockSize = 0;
    info.pAllocationCallbacks = nullptr;
    info.pDeviceMemoryCallbacks = nullptr;
    info.pHeapSizeLimit = nullptr;
    info.pVulkanFunctions = nullptr;
    info.instance = instance;
    info.vulkanApiVersion = VK_MAKE_API_VERSION(0, RequiredApiVersion::MAJOR, RequiredApiVersion::MINOR, 0);

    VmaAllocator allocator;
    if (vmaCreateAllocator(&info, &allocator) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create vma allocator."};
    }

    return allocator;
}

}  // namespace graphics
}  // namespace impl
