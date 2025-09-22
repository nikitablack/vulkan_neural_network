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

auto calculate_output_delta(GraphicsManager& graphicsManager,  //
                            VkCommandBuffer commandBuffer,  //
                            VkDescriptorSet descriptorSet,  //
                            uint32_t neuronCount,  //
                            DeviceBuffer const& values,  //
                            DeviceBuffer const& expectedOutput,  //
                            DeviceBuffer const& delta,  //
                            uint32_t expectedOutputSize,  //
                            uint32_t expectedOutputDataIndex  //
                            ) -> void;

}  // namespace graphics
}  // namespace impl
