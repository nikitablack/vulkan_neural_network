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

auto calculate_hidden_delta(GraphicsManager& graphicsManager,  //
                            VkCommandBuffer commandBuffer,  //
                            VkDescriptorSet descriptorSet,  //
                            uint32_t neuronCount,  //
                            uint32_t neighbourLayerNeuronCount,  //
                            DeviceBuffer const& neighbourWeights,  //
                            DeviceBuffer const& values,  //
                            DeviceBuffer const& neighbourDelta,  //
                            DeviceBuffer const& delta  //
                            ) -> void;

}  // namespace graphics
}  // namespace impl
