#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace impl {
namespace graphics {

class GraphicsManager;
class DeviceBuffer;

}  // namespace graphics
}  // namespace impl

namespace impl {
namespace graphics {
namespace utils {

auto read_float_helper(GraphicsManager& graphicsManager,  //
                       DeviceBuffer const& buffer,  //
                       VkPipelineStageFlags srcStageMask,  //
                       VkAccessFlags srcAccessMask,  //
                       std::vector<float>& out  // this vector should be resized to the desired size
                       ) -> void;

}  // namespace utils
}  // namespace graphics
}  // namespace impl
