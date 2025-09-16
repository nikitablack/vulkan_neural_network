#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto get_compute_queue(VkDevice device,  //
                       uint32_t queueFamilyIndex,  //
                       uint32_t queueIndex  //
                       ) noexcept -> VkQueue;

}  // namespace graphics
}  // namespace impl
