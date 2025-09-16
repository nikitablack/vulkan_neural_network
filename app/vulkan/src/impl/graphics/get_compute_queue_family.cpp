#include <impl/graphics/get_compute_queue_family.hpp>
#include <stdexcept>
#include <vector>

namespace impl {
namespace graphics {

auto get_compute_queue_family(VkPhysicalDevice device, uint32_t requiredQueueCount) -> uint32_t {
    uint32_t queueFamilyCount{0};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i{0}; i < queueFamilies.size(); ++i) {
        auto const family{queueFamilies[i]};

        if (family.queueCount >= requiredQueueCount && (family.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            return i;
        }
    }

    throw std::runtime_error{"Failed to find graphics queue."};
}

}  // namespace graphics
}  // namespace impl
