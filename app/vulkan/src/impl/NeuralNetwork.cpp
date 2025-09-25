#include <fmt/core.h>

#include <algorithm>
#include <common/Timer.hpp>
#include <impl/NeuralNetwork.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/allocate_descriptor_set.hpp>
#include <impl/graphics/calculate_hidden_delta.hpp>
#include <impl/graphics/calculate_output_delta.hpp>
#include <impl/graphics/get_command_buffer.hpp>
#include <impl/graphics/set_input_and_target_data.hpp>
#include <impl/graphics/submit.hpp>
#include <impl/graphics/update_current_batch_index.hpp>
#include <impl/graphics/utils/init_helper.hpp>
#include <impl/graphics/utils/read_float_helper.hpp>
#include <impl/graphics/utils/update_descriptor_set.hpp>
#include <random>
#include <stdexcept>

namespace {

auto shuffle_indices(std::vector<uint32_t>& indices) noexcept -> void {
    std::mt19937 engine{100};
    std::shuffle(indices.begin(), indices.end(), engine);
}

}  // namespace

namespace impl {

NeuralNetwork::NeuralNetwork(graphics::GraphicsManager& graphicsManager,  //
                             std::vector<size_t> const& layerSizes,  //
                             common::LCG<float>& lcg  //
) {
    if (layerSizes.size() < 2) {
        throw std::runtime_error{"NeuralNetwork must have at least an input and an output layer."};
    }

    layers.emplace_back(graphicsManager, layerSizes[0], 0, lcg);  // Input layer

    for (size_t i{1}; i < layerSizes.size(); ++i) {
        layers.emplace_back(graphicsManager, layerSizes[i], layerSizes[i - 1], lcg);
    }

    // Batch index - used for training.
    m_currBatchIndex.init(graphicsManager.allocator,  //
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                          2 * sizeof(uint32_t));
    graphicsManager.debugUtils.setName(m_currBatchIndex.getBuffer(), "Current batch index.");

    // Zero batch index - used for infer.
    {
        m_zeroBatchIndex.init(graphicsManager.allocator,  //
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                              2 * sizeof(uint32_t));
        graphicsManager.debugUtils.setName(m_zeroBatchIndex.getBuffer(), "Zero batch index.");

        uint32_t resetData[]{0, 0};

        graphics::utils::init_buffer_sync(graphicsManager,  //
                                          m_zeroBatchIndex,  //
                                          reinterpret_cast<uint8_t const*>(resetData),
                                          2 * sizeof(uint32_t),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_READ_BIT,  //
                                          graphicsManager.computeQueue);
    }

    // see batch_index.comp
    m_batchIndexDescriptorSet = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                  graphicsManager.descriptorPool,  //
                                                                  graphicsManager.descriptorSetLayout);
    graphicsManager.debugUtils.setName(m_batchIndexDescriptorSet, "Batch index descriptor set.");

    // see delta.comp
    m_outputDeltaDescriptorSet = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                   graphicsManager.descriptorPool,  //
                                                                   graphicsManager.descriptorSetLayout);
    graphicsManager.debugUtils.setName(m_outputDeltaDescriptorSet, "Output delta descriptor set.");

    // see delta.comp
    m_hiddenDeltaDescriptorSets.resize(layerSizes.size());
    for (size_t i{0}; i < m_hiddenDeltaDescriptorSets.size(); ++i) {
        // can throw
        m_hiddenDeltaDescriptorSets[i] = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                           graphicsManager.descriptorPool,  //
                                                                           graphicsManager.descriptorSetLayout);
        graphicsManager.debugUtils.setName(m_hiddenDeltaDescriptorSets[i],
                                           fmt::format("Hidden delta descriptor set {}.", i));
    }
}

auto NeuralNetwork::infer(graphics::GraphicsManager& graphicsManager,  //
                          std::vector<float> const& inputValues,  //
                          std::vector<float>& outputValues  //
                          ) -> void {
    // init buffer with data
    {
        auto& inputLayer{layers.front()};

        if (inputValues.size() != inputLayer.size()) {
            throw std::runtime_error{"Mismatch in input size."};
        }

        graphics::utils::init_buffer_sync(graphicsManager,  //
                                          inputLayer.values,  //
                                          reinterpret_cast<uint8_t const*>(inputValues.data()),  //
                                          inputValues.size() * sizeof(float),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_WRITE_BIT,  //
                                          graphicsManager.computeQueue);
    }

    auto const commandBuffer{graphics::get_command_buffer_begin(graphicsManager.device, graphicsManager.commandPool)};

    forward(graphicsManager, commandBuffer, m_zeroBatchIndex, true);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to end copy command buffer."};
    }

    graphics::submit(commandBuffer, graphicsManager.computeQueue.queue, VK_NULL_HANDLE);

    if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to wait queue on infer."};
    }

    auto const& outputLayer{layers.back()};

    outputValues.resize(outputLayer.size());

    graphics::utils::read_float_helper(graphicsManager,  //
                                       outputLayer.values,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                       VK_ACCESS_SHADER_WRITE_BIT,  //
                                       outputValues);
}

auto NeuralNetwork::train(graphics::GraphicsManager& graphicsManager,  //
                          std::vector<std::vector<float>> const& input,  // 0.0-1.0
                          std::vector<uint8_t> const& target,  // 0-9
                          size_t epochCount,  //
                          float learningRate  //
                          ) -> void {
    // assuming that all the inputs are of equal size
    if ((input.size() == 0) || (input[0].size() == 0)) {
        throw std::runtime_error{"Empty input."};
    }

    if (input.size() != target.size()) {
        throw std::runtime_error{"Mismatch between input size and target size."};
    }

    // Init buffers with the train data.
    // Upload all the data at once to avoid repeated GPU uploads.
    graphics::set_input_and_target_data(graphicsManager,  //
                                        layers.front(),  //
                                        input,  //
                                        layers.back(),  //
                                        m_expectedOutput,  //
                                        target);

    // see delta.comp
    auto const outputLayer{layers.back()};

    graphics::utils::update_descriptor_set(
        graphicsManager.device,  //
        m_outputDeltaDescriptorSet,  //
        outputLayer.values,  // dummy, not used for for this set, but have to be provided (without specifying a the
                             // `nullDescriptor` feature)
        outputLayer.values,  //
        m_expectedOutput,  //
        outputLayer.delta,  //
        m_currBatchIndex);

    // Indices for shuffling. In the beginning it will be filled with increasing sequence, i.e. 0, 1, 2, 3, etc.
    // After each epoch it will be shuffled, so the training is different
    std::vector<uint32_t> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Lazy initialization of the indices buffer.
    if (!m_batchIndices.getBuffer()) {
        m_batchIndices.init(graphicsManager.allocator,  //
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                            indices.size() * sizeof(uint32_t));
        graphicsManager.debugUtils.setName(m_batchIndices.getBuffer(), "Batch indices.");

        // update these sets only once, since they never change afterwards
        //
        // see batch_index.comp
        graphics::utils::update_descriptor_set(graphicsManager.device,  //
                                               m_batchIndexDescriptorSet,  //
                                               m_currBatchIndex,  //
                                               m_batchIndices);
    }

    if (!m_trainCommandBuffer) {
        m_trainCommandBuffer = createTrainCommandBuffer(graphicsManager, learningRate);
    }

    common::Timer totalTimer{};
    totalTimer.start();

    for (size_t epoch{0}; epoch < epochCount; ++epoch) {
        fmt::println("epoch {}", epoch);

        // For each epoch, the indices are different.
        // We need to upload upload the to the GPU each epoch.
        {
            graphics::utils::init_buffer_sync(graphicsManager,  //
                                              m_batchIndices,  //
                                              reinterpret_cast<uint8_t const*>(indices.data()),
                                              indices.size() * sizeof(uint32_t),  //
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                              VK_ACCESS_SHADER_READ_BIT,  //
                                              graphicsManager.computeQueue);
        }

        // Reset batch index.
        // After reset, the index will point to the 0-th element of the m_batchIndices.
        {
            uint32_t resetData[]{0, 0};

            graphics::utils::init_buffer_sync(graphicsManager,  //
                                              m_currBatchIndex,  //
                                              reinterpret_cast<uint8_t const*>(resetData),
                                              2 * sizeof(uint32_t),  //
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,  //
                                              graphicsManager.computeQueue);
        }

        // Submit the same command buffer multiple times.
        for (size_t i{0}; i < indices.size(); ++i) {
            graphics::submit(m_trainCommandBuffer, graphicsManager.computeQueue.queue, VK_NULL_HANDLE);
        }

        if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
            throw std::runtime_error{"Failed to wait queue on train."};
        }

        shuffle_indices(indices);
    }

    auto const totalTimeMs{totalTimer.stop()};
    fmt::println("Training completed in {:.2f} ms", totalTimeMs);
    fmt::println("Average epoch time: {:.2f} ms", totalTimeMs / epochCount);
}

auto NeuralNetwork::forward(graphics::GraphicsManager& graphicsManager,  //
                            VkCommandBuffer commandBuffer,  //
                            graphics::DeviceBuffer const& batchIndexBuffer,  //
                            bool infer  //
                            ) -> void {
    for (size_t i{1}; i < layers.size(); ++i) {
        auto& currLayer{layers[i]};
        auto& prevLayer{layers[i - 1]};

        currLayer.activate(graphicsManager,  //
                           commandBuffer,  //
                           prevLayer,  //
                           batchIndexBuffer,  //
                           infer);
    }
}

auto NeuralNetwork::backward(graphics::GraphicsManager& graphicsManager,  //
                             VkCommandBuffer commandBuffer,  //
                             float learningRate,  //
                             graphics::DeviceBuffer const& batchIndexBuffer  //
                             ) -> void {
    auto& outputLayer{layers.back()};

    // calculate deltas
    {
        // special case - output layer
        graphics::calculate_output_delta(graphicsManager,  //
                                         commandBuffer,  //
                                         m_outputDeltaDescriptorSet,  //
                                         outputLayer.size(),  //
                                         outputLayer.values);

        // the hidden layers
        for (size_t layerInd{layers.size() - 2}; layerInd > 0; --layerInd) {
            auto& layer{layers[layerInd]};
            auto const& rightLayer{layers[layerInd + 1]};

            graphics::calculate_hidden_delta(graphicsManager,  //
                                             commandBuffer,  //
                                             m_hiddenDeltaDescriptorSets[layerInd],  //
                                             layer.size(),  //
                                             rightLayer.size(),  //
                                             rightLayer.weights,  //
                                             layer.values,  //
                                             rightLayer.delta,  //
                                             layer.delta);
        }
    }

    // update layers
    {
        for (size_t layerInd{1}; layerInd < layers.size(); ++layerInd) {
            auto& layer{layers[layerInd]};
            auto const& leftLayer{layers[layerInd - 1]};

            // can throw
            layer.update(graphicsManager,  //
                         commandBuffer,  //
                         leftLayer,  //
                         learningRate,  //
                         batchIndexBuffer);
        }
    }
}

auto NeuralNetwork::clear() noexcept -> void {
    for (auto& layer : layers) {
        layer.clear();
    }

    m_expectedOutput.destroy();
    m_batchIndices.destroy();
    m_currBatchIndex.destroy();
    m_zeroBatchIndex.destroy();
}

auto NeuralNetwork::createTrainCommandBuffer(impl::graphics::GraphicsManager& graphicsManager,  //
                                             float learningRate  //
                                             ) -> VkCommandBuffer {
    VkCommandBuffer const commandBuffer{
        graphics::get_command_buffer_begin(graphicsManager.device,  //
                                           graphicsManager.commandPool,  //
                                           VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)};

    graphics::update_current_batch_index(graphicsManager,  //
                                         commandBuffer,  //
                                         m_batchIndexDescriptorSet,  //
                                         m_currBatchIndex);

    // can throw
    forward(graphicsManager, commandBuffer, m_currBatchIndex, false);

    // can throw
    backward(graphicsManager, commandBuffer, learningRate, m_currBatchIndex);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to end copy command buffer."};
    }

    return commandBuffer;
}

}  // namespace impl
