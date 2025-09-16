#include <fmt/core.h>

#include <algorithm>
#include <common/Timer.hpp>
#include <impl/NeuralNetwork.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/submit.hpp>
#include <impl/graphics/utils/init_helper.hpp>
#include <impl/graphics/utils/read_float_helper.hpp>
#include <random>
#include <stdexcept>

namespace {

auto shuffle_indices(std::vector<size_t>& indices) noexcept -> void {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
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

    // can throw
    layers.emplace_back(graphicsManager, layerSizes[0], 0, lcg);  // Input layer

    for (size_t i{1}; i < layerSizes.size(); ++i) {
        // can throw
        layers.emplace_back(graphicsManager, layerSizes[i], layerSizes[i - 1], lcg);
    }

    // can throw
    m_output.init(graphicsManager.allocator,  //
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  //
                  layerSizes.back() * sizeof(float));
}

auto NeuralNetwork::forward(graphics::GraphicsManager& graphicsManager,  //
                            std::vector<float> const& inputValues,  //
                            std::vector<float>* outputValues  //
                            ) -> void {
    auto& inputLayer{layers.front()};

    if (inputValues.size() != inputLayer.size()) {
        throw std::runtime_error{"Mesmatch in input size."};
    }

    auto const commandBuffer{graphicsManager.commandManager.getCommandBufferBegin()};

    auto stagingBufferInput{graphics::utils::init_buffer(commandBuffer,  //
                                                         inputLayer.values,  //
                                                         reinterpret_cast<uint8_t const*>(inputValues.data()),  //
                                                         inputValues.size() * sizeof(float),  //
                                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                                         VK_ACCESS_SHADER_READ_BIT)};

    for (size_t i{1}; i < layers.size(); ++i) {
        auto& currLayer{layers[i]};
        auto& prevLayer{layers[i - 1]};

        currLayer.activate(graphicsManager, commandBuffer, prevLayer);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to end copy command buffer."};
    }

    graphics::submit(commandBuffer, graphicsManager.computeQueue.queue);

    if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to wait staging queue."};
    }

    stagingBufferInput.destroy();

    if (outputValues) {
        auto const& outputLayer{layers.back()};

        outputValues->resize(outputLayer.size());

        graphics::utils::read_float_helper(graphicsManager,  //
                                           outputLayer.values,
                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                           VK_ACCESS_SHADER_WRITE_BIT,  //
                                           *outputValues);
    }
}

auto NeuralNetwork::train(graphics::GraphicsManager& graphicsManager,  //
                          std::vector<std::vector<float>> const& input,  // 0.0-1.0
                          std::vector<uint8_t> const& target,  // 0-9
                          size_t epochCount,  //
                          float learningRate  //
                          ) -> void {
    if (input.size() != target.size()) {
        throw std::runtime_error{"Mismatch between input size and target size."};
    }

    std::vector<size_t> indices(input.size());  // Indices for shuffling
    std::iota(indices.begin(), indices.end(), 0);

    common::Timer totalTimer{};

    totalTimer.start();

    for (size_t epoch{0}; epoch < epochCount; ++epoch) {
        // float epochLoss{0.0};

        for (size_t i{0}; i < indices.size(); ++i) {
            auto const idx{indices[i]};

            // can throw
            forward(graphicsManager, input[idx], nullptr);

            auto const& outputLayer{layers.back()};

            std::vector<float> expectedOutput(outputLayer.size(), 0.0);
            expectedOutput[target[idx]] = 1.0;

            // float loss{0.0f};
            // for (size_t j{0}; j < output.size(); ++j) {
            //     float const diff{expectedOutput[j] - output[j]};
            //     loss += diff * diff;
            // }
            // loss /= output.size();  // Mean squared error

            // epochLoss += loss;

            // can throw
            backward(graphicsManager, expectedOutput, learningRate);
        }

        shuffle_indices(indices);

        // float const averageLoss{epochLoss / input.size()};
        // fmt::println("Epoch {}:\n\taverage loss: {}\n\tepoch time: {:.2f} ms", epoch, averageLoss,
        // epochTimer.stop());
    }

    auto const totalTimeMs{totalTimer.stop()};
    fmt::println("Training completed in {:.2f} ms", totalTimeMs);
    fmt::println("Average epoch time: {:.2f} ms", totalTimeMs / epochCount);
}

auto NeuralNetwork::backward(graphics::GraphicsManager& /* graphicsManager */,  //
                             std::vector<float> const& expectedOutput,  //
                             float /* learningRate */  //
                             ) -> void {
    if (m_output.getSize() != expectedOutput.size() * sizeof(float)) {
        throw std::runtime_error{"Mismatch in output size."};
    }

    // auto const commandBuffer{graphicsManager.commandManager.getCommandBufferBegin()};

    // auto& outputLayer{layers.back()};

    // calculate output delta
    // Layer::MatrixX deltaOutput{output.size(), 1};
    // for (size_t i{0}; i < output.size(); ++i) {
    //     Float const a{output[i]};
    //     Float const dCdA{2.0_F * (a - expectedOutput[i]) / output.size()};
    //     Float const dAdZ{sigmoidDerivative(a)};
    //     deltaOutput(static_cast<Eigen::Index>(i), 0) = dCdA * dAdZ;
    // }

    {
        // auto const& leftLayer{layers[layers.size() - 2]};

        // // can throw
        // outputLayer.update(graphicsManager, commandBuffer, leftLayer, learningRate, m_delta);
    }

    // update hidden layers
    // {
    //     auto delta{std::move(deltaOutput)};

    //     auto const* rightLayer{&outputLayer};

    //     for (size_t lay{layers.size() - 2}; lay > 0; --lay) {
    //         auto& currLayer{layers[lay]};
    //         auto const& leftLayer{layers[lay - 1]};

    //         Layer::MatrixX const WT{rightLayer->weights.transpose()};
    //         Layer::MatrixX WTD{WT * delta};
    //         Layer::MatrixX dZ{currLayer.values.unaryExpr(&sigmoidDerivative)};
    //         Layer::MatrixX deltaHidden{WTD.cwiseProduct(dZ)};

    //         if (!currLayer.update(leftLayer, learningRate, deltaHidden)) {
    //             return false;
    //         }

    //         // Prepare for the next iteration
    //         delta = std::move(deltaHidden);
    //         rightLayer = &currLayer;
    //     }
    // }
}

auto NeuralNetwork::clear() noexcept -> void {
    for (auto& layer : layers) {
        layer.clear();
    }
}

}  // namespace impl
