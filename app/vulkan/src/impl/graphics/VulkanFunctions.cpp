#include <fmt/core.h>

#include <impl/graphics/VulkanFunctions.hpp>
#include <stdexcept>

namespace {

template <typename F>
auto get_function(VkInstance instance, char const* name) -> F {
    auto const f{reinterpret_cast<F>(vkGetInstanceProcAddr(instance, name))};

    if (!f) {
        throw std::runtime_error{fmt::format("Failed to get function {}.", name)};
    }

    return f;
}

}  // namespace

namespace impl {
namespace graphics {

// debug utils
PFN_vkSetDebugUtilsObjectNameEXT VulkanFunctions::vkSetDebugUtilsObjectNameEXT{VK_NULL_HANDLE};

auto VulkanFunctions::initialize(VkInstance instance) -> void {
    // debug utils
    vkSetDebugUtilsObjectNameEXT =
        get_function<PFN_vkSetDebugUtilsObjectNameEXT>(instance, "vkSetDebugUtilsObjectNameEXT");
}

}  // namespace graphics
}  // namespace impl
