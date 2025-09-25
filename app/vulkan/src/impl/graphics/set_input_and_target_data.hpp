#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace impl {

class Layer;

}

namespace impl {
namespace graphics {

class GraphicsManager;

}
}  // namespace impl

namespace impl {
namespace graphics {

auto set_input_and_target_data(GraphicsManager& graphicsManager,  //
                               Layer& inputLayer,  //
                               std::vector<std::vector<float>> const& input,  //
                               Layer& outputLayer,  //
                               DeviceBuffer& expectedOutput,  //
                               std::vector<uint8_t> const& target  //
                               ) -> void;

}  // namespace graphics
}  // namespace impl