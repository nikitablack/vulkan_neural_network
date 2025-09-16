#pragma once

#include <vulkan/vulkan.h>

#include <string>

namespace impl {
namespace graphics {

auto create_shader_module(VkDevice device, std::string const& name) -> VkShaderModule;

}  // namespace graphics
}  // namespace impl
