#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

struct VulkanQueue {
public:
    uint32_t queueFamily;
    VkQueue queue;
};

}  // namespace graphics
}  // namespace impl
