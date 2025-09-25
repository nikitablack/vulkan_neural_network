#pragma once

#include <vulkan/vulkan.h>

namespace impl {
namespace graphics {

class DeviceBuffer;
class GraphicsManager;

}  // namespace graphics
}  // namespace impl

namespace impl {
namespace graphics {

auto update_current_batch_index(GraphicsManager& graphicsManager,  //
                                VkCommandBuffer commandBuffer,  //
                                VkDescriptorSet descriptorSet,  //
                                DeviceBuffer const& currBatchIndex  //
                                ) -> void;

}  // namespace graphics
}  // namespace impl
