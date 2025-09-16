#include <impl/graphics/VulkanDebugUtils.hpp>
#include <impl/graphics/get_compute_queue.hpp>

namespace impl {
namespace graphics {

auto get_compute_queue(VkDevice device,  //
                       uint32_t queueFamilyIndex,  //
                       uint32_t queueIndex  //
                       ) noexcept -> VkQueue {
    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, &queue);

    return queue;
}

}  // namespace graphics
}  // namespace impl
