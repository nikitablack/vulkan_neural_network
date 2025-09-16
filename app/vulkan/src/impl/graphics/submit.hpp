#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto submit(VkCommandBuffer commandBuffer, VkQueue queue) -> void;

}  // namespace graphics
}  // namespace impl