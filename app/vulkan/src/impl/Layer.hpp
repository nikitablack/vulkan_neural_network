#pragma once

#include <impl/graphics/DeviceBuffer.hpp>
#include <vector>

namespace common {

template <typename T>
class LCG;

}

namespace impl {
namespace graphics {

class GraphicsManager;

}
}  // namespace impl

namespace impl {

class Layer {
public:
    Layer() noexcept = default;

    Layer(graphics::GraphicsManager& graphicsManager,  //
          size_t neuronCount,  //
          size_t inputCount,  //
          common::LCG<float>& lcg  //
    );

public:
    auto activate(graphics::GraphicsManager const& graphicsManager,  //
                  VkCommandBuffer commandBuffer,  //
                  Layer const& prevLayer  //
                  ) -> void;

    auto update(graphics::GraphicsManager const& graphicsManager,  //
                VkCommandBuffer commandBuffer,  //
                Layer const& prevLayer,  //
                float learningRate,  //
                graphics::DeviceBuffer const& delta  //
                ) -> void;

    auto size() const noexcept -> size_t;
    auto inputSize() const noexcept -> size_t;
    auto clear() noexcept -> void;

public:
    graphics::DeviceBuffer weights{};
    graphics::DeviceBuffer biases{};
    graphics::DeviceBuffer values{};

private:
    size_t m_size{};
    size_t m_inputSize{};
};

}  // namespace impl
