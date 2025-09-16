#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {
namespace features {

struct Features {
    static auto print() noexcept -> void;
    static auto check(VkPhysicalDevice physicalDevice) noexcept -> bool;
    auto get_chain() noexcept -> VkPhysicalDeviceFeatures2*;
};

}  // namespace features
}  // namespace graphics
}  // namespace impl
