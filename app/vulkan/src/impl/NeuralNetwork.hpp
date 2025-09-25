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

    auto clear() noexcept -> void;

private:
    auto forward(graphics::GraphicsManager& graphicsManager,  //
                 VkCommandBuffer commandBuffer,  //
                 graphics::DeviceBuffer const& batchIndexBuffer,  //
                 bool infer = false  //
                 ) -> void;

    auto backward(graphics::GraphicsManager& graphicsManager,  //
                  VkCommandBuffer commandBuffer,  //
                  float learningRate,  //
                  graphics::DeviceBuffer const& batchIndexBuffer  //
                  ) -> void;

    auto createTrainCommandBuffer(impl::graphics::GraphicsManager& graphicsManager,  //
                                  float learningRate  //
                                  ) -> VkCommandBuffer;

private:
    std::vector<Layer> layers;
    graphics::DeviceBuffer m_expectedOutput{};
    graphics::DeviceBuffer m_batchIndices{};
    graphics::DeviceBuffer m_currBatchIndex{};
    graphics::DeviceBuffer m_zeroBatchIndex{};  // used for inference
    VkDescriptorSet m_batchIndexDescriptorSet{VK_NULL_HANDLE};
    VkDescriptorSet m_outputDeltaDescriptorSet{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> m_hiddenDeltaDescriptorSets{};
    VkCommandBuffer m_trainCommandBuffer{VK_NULL_HANDLE};
};

}  // namespace impl
