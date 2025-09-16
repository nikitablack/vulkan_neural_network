#include <fmt/core.h>

#include <impl/graphics/RequiredDeviceExtensions.hpp>
#include <impl/graphics/check_physical_device_support.hpp>
#include <impl/graphics/features/Features.hpp>
#include <impl/graphics/get_physical_device_properties.hpp>
#include <impl/graphics/get_supported_physical_devices.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto get_supported_physical_devices(VkInstance instance) -> std::vector<VkPhysicalDevice> {
    RequiredDeviceExtensions::print();
    features::Features::print();

    uint32_t deviceCount{0};

    if (vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0) {
        throw std::runtime_error{"Failed to find GPUs with Vulkan support."};
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);

    if (vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to find GPUs with Vulkan support."};
    }

    std::vector<VkPhysicalDevice> supportedPhysicalDevices{};
    supportedPhysicalDevices.reserve(deviceCount);

    for (auto const device : physicalDevices) {
        auto const props{get_physical_device_properties(device)};

        try {
            check_physical_device_support(device);

            supportedPhysicalDevices.push_back(device);

            fmt::println("{} is supported.", props.deviceName);
        } catch (std::exception& e) {
            fmt::println("{} is not supported: {}", props.deviceName, e.what());
        }
    }

    if (supportedPhysicalDevices.empty()) {
        throw std::runtime_error{"Failed to find supported device."};
    }

    fmt::println("The number of supported devices: {}", supportedPhysicalDevices.size());

    return supportedPhysicalDevices;
}

}  // namespace graphics
}  // namespace impl
