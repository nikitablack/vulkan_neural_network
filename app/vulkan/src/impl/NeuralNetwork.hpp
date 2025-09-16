#pragma once

#include <impl/Layer.hpp>

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

class NeuralNetwork {
public:
    NeuralNetwork(graphics::GraphicsManager& graphicsManager,  //
                  std::vector<size_t> const& layerSizes,  //
                  common::LCG<float>& lcg  //
    );

    auto forward(graphics::GraphicsManager& graphicsManager,  //
                 std::vector<float> const& inputValues,  //
                 std::vector<float>* outputValues = nullptr  //
                 ) -> void;

    auto train(graphics::GraphicsManager& graphicsManager,  //
               std::vector<std::vector<float>> const& input,  //
               std::vector<uint8_t> const& target,  //
               size_t epochCount,  //
               float learningRate  //
               ) -> void;

    auto clear() noexcept -> void;

private:
    auto backward(graphics::GraphicsManager& graphicsManager,  //
                  std::vector<float> const& expectedOutput,  //
                  float learningRate  //
                  ) -> void;

public:
    std::vector<Layer> layers;
    graphics::DeviceBuffer m_output{};
    graphics::HostVisibleBuffer m_expectedOutput{};
};

}  // namespace impl
