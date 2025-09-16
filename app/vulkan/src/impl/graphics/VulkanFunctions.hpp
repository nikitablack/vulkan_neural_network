#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

struct VulkanFunctions {
public:
    static auto initialize(VkInstance instance) -> void;

public:
    // debug utils
    static PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
};

}  // namespace graphics
}  // namespace impl
