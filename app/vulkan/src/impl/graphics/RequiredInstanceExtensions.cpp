#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <impl/graphics/RequiredInstanceExtensions.hpp>

namespace impl {
namespace graphics {

auto RequiredInstanceExtensions::get() noexcept -> const std::vector<std::string>& {
    static std::vector<std::string> extensions{};

    if (extensions.empty()) {
#ifdef ENABLE_VULKAN_DEBUG_UTILS
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    }

    return extensions;
}

auto RequiredInstanceExtensions::print() noexcept -> void {
    fmt::print("Required instance extensions:\n");

    for (auto const& ext : get()) {
        fmt::print("\t{}\n", ext);
    }
}

}  // namespace graphics
}  // namespace impl
