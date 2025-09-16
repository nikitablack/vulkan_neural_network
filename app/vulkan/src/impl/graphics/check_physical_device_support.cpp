#include <fmt/core.h>

#include <impl/graphics/RequiredApiVersion.hpp>
#include <impl/graphics/RequiredDeviceExtensions.hpp>
#include <impl/graphics/check_physical_device_support.hpp>
#include <impl/graphics/features/Features.hpp>
#include <impl/graphics/get_physical_device_properties.hpp>
#include <stdexcept>
#include <vector>

namespace {

auto check_required_device_extensions(VkPhysicalDevice device) -> void {
    uint32_t extensionCount{0};

    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to get physical device extension properties."};
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);

    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data()) !=
        VK_SUCCESS) {
        throw std::runtime_error{"Failed to get physical device extension properties."};
    }

    for (auto const& reqExt : impl::graphics::RequiredDeviceExtensions::get()) {
        for (auto const& avExt : availableExtensions) {
            if (reqExt == avExt.extensionName) {
                goto cnt;
            }
        }

        throw std::runtime_error{"Required device extension \"" + reqExt + "\" is not supported."};

    cnt:;
    }
}

}  // namespace

namespace impl {
namespace graphics {

auto check_physical_device_support(VkPhysicalDevice device) -> void {
    const auto props{get_physical_device_properties(device)};

    // api version
    const uint32_t major{VK_API_VERSION_MAJOR(props.apiVersion)};
    const uint32_t minor{VK_API_VERSION_MINOR(props.apiVersion)};
    const uint32_t patch{VK_API_VERSION_PATCH(props.apiVersion)};

    if (major < RequiredApiVersion::MAJOR ||
        (major == RequiredApiVersion::MAJOR && minor < RequiredApiVersion::MINOR)) {
        throw std::runtime_error{
            fmt::format("Minimum supported Vulkan api version is {}.{}.0. The device's api version is {}.{}.{}.",
                        RequiredApiVersion::MAJOR, RequiredApiVersion::MINOR, major, minor, patch)};
    }

    if (!features::Features::check(device)) {
        throw std::runtime_error{"Required features are not supported."};
    }

    check_required_device_extensions(device);
}

}  // namespace graphics
}  // namespace impl
