#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <impl/graphics/RequiredDeviceExtensions.hpp>

namespace impl {
namespace graphics {

auto RequiredDeviceExtensions::get() noexcept -> std::vector<std::string> const& {
    static std::vector<std::string> extensions{};

    if (extensions.empty()) {
        // extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }

    return extensions;
}

auto RequiredDeviceExtensions::print() noexcept -> void {
    fmt::println("Required device extensions:");

    for (auto const& ext : get()) {
        fmt::println("\t{}", ext);
    }
}

}  // namespace graphics
}  // namespace impl
