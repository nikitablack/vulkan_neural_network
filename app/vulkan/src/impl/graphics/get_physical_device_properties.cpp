#include <impl/graphics/get_physical_device_properties.hpp>

namespace impl {
namespace graphics {

auto get_physical_device_properties(VkPhysicalDevice device) noexcept -> VkPhysicalDeviceProperties {
    VkPhysicalDeviceProperties physicalDeviceProperties{};

    vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

    return physicalDeviceProperties;
}

}  // namespace graphics
}  // namespace impl
