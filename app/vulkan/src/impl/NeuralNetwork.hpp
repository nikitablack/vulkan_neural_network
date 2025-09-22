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

    auto infer(graphics::GraphicsManager& graphicsManager,  //
               std::vector<float> const& inputValues,  //
               std::vector<float>& outputValues  //
               ) -> void;

    auto train(graphics::GraphicsManager& graphicsManager,  //
               std::vector<std::vector<float>> const& input,  //
               std::vector<uint8_t> const& target,  //
               size_t epochCount,  //
               float learningRate  //
               ) -> void;

    auto clear(graphics::GraphicsManager const& graphicsManager) noexcept -> void;

private:
    auto forward(graphics::GraphicsManager& graphicsManager,  //
                 VkCommandBuffer commandBuffer,  //
                 uint32_t dataIndex,  //
                 uint32_t iterationIndex  //
                 ) -> void;

    auto backward(graphics::GraphicsManager& graphicsManager,  //
                  VkCommandBuffer commandBuffer,  //
                  uint32_t dataIndex,  //
                  float learningRate,  //
                  uint32_t iterationIndex  //
                  ) -> void;

public:
    std::vector<Layer> layers;
    graphics::DeviceBuffer m_expectedOutput{};
    uint32_t m_currIteration{0};
    std::array<VkFence, graphics::CONCURRENT_ITERATIONS_COUNT> m_fences{};
    std::array<VkDescriptorSet, graphics::CONCURRENT_ITERATIONS_COUNT> m_outputDeltaDescriptorSets{};
    std::array<std::vector<VkDescriptorSet>, graphics::CONCURRENT_ITERATIONS_COUNT> m_hiddenDeltaDescriptorSets{};
};

}  // namespace impl
