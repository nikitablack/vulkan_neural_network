#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

auto create_instance() -> VkInstance;

}  // namespace graphics
}  // namespace impl
