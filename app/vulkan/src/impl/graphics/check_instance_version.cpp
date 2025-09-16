#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <impl/graphics/RequiredApiVersion.hpp>
#include <impl/graphics/check_instance_version.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto check_instance_version() -> void {
    if ((RequiredApiVersion::MAJOR > 1) || (RequiredApiVersion::MAJOR == 1 && RequiredApiVersion::MINOR > 1)) {
        auto const f{reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"))};

        if (!f) {
            throw std::runtime_error{"Your version of Vulkan is < 1.1. Please update the graphics driver."};
        }

        uint32_t apiVersion;
        if (vkEnumerateInstanceVersion(&apiVersion) != VK_SUCCESS) {
            throw std::runtime_error{"Failed to enumerate instance version."};
        }

        uint32_t const major{VK_API_VERSION_MAJOR(apiVersion)};
        uint32_t const minor{VK_API_VERSION_MINOR(apiVersion)};
        uint32_t const patch{VK_API_VERSION_PATCH(apiVersion)};

        fmt::println("Instance version: {}.{}.{}", major, minor, patch);

        if (major < RequiredApiVersion::MAJOR ||
            (major == RequiredApiVersion::MAJOR && minor < RequiredApiVersion::MINOR)) {
            throw std::runtime_error{
                fmt::format("Minimum supported Vulkan api version is {}.{}.0. Your version of Vulkan is {}.{}.{}.",
                            RequiredApiVersion::MAJOR, RequiredApiVersion::MINOR, major, minor, patch)};
        }
    }
}

}  // namespace graphics
}  // namespace impl
