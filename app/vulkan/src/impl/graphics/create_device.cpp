#include <algorithm>
#include <impl/graphics/RequiredDeviceExtensions.hpp>
#include <impl/graphics/create_device.hpp>
#include <impl/graphics/features/Features.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_device(VkPhysicalDevice physicalDevice, uint32_t graphicsQueueFamily, uint32_t queueCount) -> VkDevice {
    // priorities
    std::vector<float> queuePriorities(queueCount);
    std::fill(queuePriorities.begin(), queuePriorities.end(), 1.0f);

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pNext = nullptr;
    queueCreateInfo.flags = 0;
    queueCreateInfo.queueFamilyIndex = graphicsQueueFamily;
    queueCreateInfo.queueCount = queueCount;
    queueCreateInfo.pQueuePriorities = queuePriorities.data();

    features::Features features{};

    auto const requiredExtensions{RequiredDeviceExtensions::get()};

    std::vector<char const*> extensionsStr{};
    extensionsStr.reserve(requiredExtensions.size());

    for (auto const& extension : requiredExtensions) {
        extensionsStr.push_back(extension.data());
    }

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = features.get_chain();
    deviceCreateInfo.flags = 0;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensionsStr.size());
    deviceCreateInfo.ppEnabledExtensionNames = extensionsStr.data();
    deviceCreateInfo.pEnabledFeatures = nullptr;

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create logical device."};
    }

    return device;
}

}  // namespace graphics
}  // namespace impl
