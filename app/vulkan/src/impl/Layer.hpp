#pragma once

#include <array>
#include <impl/graphics/DeviceBuffer.hpp>
#include <unordered_map>
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
                  Layer const& prevLayer,  //
                  graphics::DeviceBuffer const& batchIndex,  //
                  bool infer = false  //
                  ) -> void;

    auto update(graphics::GraphicsManager const& graphicsManager,  //
                VkCommandBuffer commandBuffer,  //
                Layer const& prevLayer,  //
                float learningRate,  //
                graphics::DeviceBuffer const& batchIndex  //
                ) -> void;

    auto size() const noexcept -> size_t;
    auto inputSize() const noexcept -> size_t;
    auto clear() noexcept -> void;

public:
    graphics::DeviceBuffer weights{};
    graphics::DeviceBuffer biases{};
    graphics::DeviceBuffer values{};
    graphics::DeviceBuffer delta{};

private:
    size_t m_size{};
    size_t m_inputSize{};

    // graphics
    VkDescriptorSet m_activateDescriptorSet{VK_NULL_HANDLE};
    bool m_activateDescriptorSetUpdated{false};

    VkDescriptorSet m_inferDescriptorSet{VK_NULL_HANDLE};
    bool m_inferDescriptorSetUpdated{false};

    VkDescriptorSet m_updateDescriptorSet{VK_NULL_HANDLE};
    bool m_updateDescriptorSetUpdated{false};
};

}  // namespace impl
