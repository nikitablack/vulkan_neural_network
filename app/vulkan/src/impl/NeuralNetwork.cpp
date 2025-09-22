#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <common/Timer.hpp>
#include <impl/NeuralNetwork.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/allocate_descriptor_set.hpp>
#include <impl/graphics/calculate_hidden_delta.hpp>
#include <impl/graphics/calculate_output_delta.hpp>
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

    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i{0}; i < m_fences.size(); ++i) {
        if (vkCreateFence(graphicsManager.device, &info, nullptr, &m_fences[i]) != VK_SUCCESS) {
            throw std::runtime_error{fmt::format("Failed to create NeuralNetwork fence {}.", i)};
        }

        graphicsManager.debugUtils.setName(m_fences[i], fmt::format("NeuralNetwork fence {}.", i));
    }

    for (size_t i{0}; i < m_outputDeltaDescriptorSets.size(); ++i) {
        m_outputDeltaDescriptorSets[i] = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                           graphicsManager.descriptorPool,  //
                                                                           graphicsManager.descriptorSetLayout);
        graphicsManager.debugUtils.setName(m_outputDeltaDescriptorSets[i],
                                           fmt::format("Output delta descriptor set {}.", i));
    }

    for (size_t i{0}; i < m_hiddenDeltaDescriptorSets.size(); ++i) {
        auto& v{m_hiddenDeltaDescriptorSets[i]};
        v.resize(layerSizes.size());

        for (size_t j{0}; j < v.size(); ++j) {
            v[j] = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                     graphicsManager.descriptorPool,  //
                                                     graphicsManager.descriptorSetLayout);
            graphicsManager.debugUtils.setName(v[j], fmt::format("Hidden delta descriptor set {}:{}.", i, j));
        }
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

        graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
                                          inputLayer.values,  //
                                          reinterpret_cast<uint8_t const*>(inputValues.data()),  //
                                          inputValues.size() * sizeof(float),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_WRITE_BIT,  //
                                          graphicsManager.computeQueue);
    }

    auto const commandBuffer{graphicsManager.commandManager.getCommandBufferBegin()};

    forward(graphicsManager, commandBuffer, 0, 0);

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
    if (input.size() == 0) {
        throw std::runtime_error{"Empty input."};
    }

    // assuming that all the inputs are of equal size
    if (input[0].size() == 0) {
        throw std::runtime_error{"Empty input."};
    }

    if (input.size() != target.size()) {
        throw std::runtime_error{"Mismatch between input size and target size."};
    }

    // init buffer with data
    // upload all inputs at once to avoid repeated GPU uploads
    {
        auto& inputLayer{layers.front()};

        auto const singleInputSizeBytes{input[0].size() * sizeof(float)};
        auto const totalInputSizeBytes{input.size() * singleInputSizeBytes};

        if (inputLayer.values.getSize() < totalInputSizeBytes) {
            if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
                throw std::runtime_error{"Failed to wait queue on train."};
            }

            inputLayer.values.destroy();

            inputLayer.values.init(graphicsManager.allocator,  //
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                                   totalInputSizeBytes);

            graphicsManager.debugUtils.setName(inputLayer.values.getBuffer(), "Layer values.");
        }

        graphics::HostVisibleBuffer stagingBuffer{graphicsManager.allocator,  //
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                                                  totalInputSizeBytes};

        for (size_t inputIdx{0}; inputIdx < input.size(); ++inputIdx) {
            auto const& v{input[inputIdx]};

            stagingBuffer.copyData(reinterpret_cast<uint8_t const*>(v.data()),  //
                                   singleInputSizeBytes,  //
                                   singleInputSizeBytes * inputIdx);
        }

        graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
                                          inputLayer.values,  //
                                          std::move(stagingBuffer),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_WRITE_BIT,  //
                                          graphicsManager.computeQueue);
    }

    // init expected output all at once
    {
        auto const& outputLayer{layers.back()};

        auto const singleOutputSizeBytes{outputLayer.size() * sizeof(float)};
        auto const totalOutputSizeBytes{target.size() * singleOutputSizeBytes};

        if (m_expectedOutput.getSize() < totalOutputSizeBytes) {
            if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
                throw std::runtime_error{"Failed to wait queue on train."};
            }

            m_expectedOutput.destroy();

            m_expectedOutput.init(graphicsManager.allocator,  //
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                                  totalOutputSizeBytes);

            graphicsManager.debugUtils.setName(m_expectedOutput.getBuffer(), "Expected output.");
        }

        graphics::HostVisibleBuffer stagingBuffer{graphicsManager.allocator,  //
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                                                  totalOutputSizeBytes};

        std::vector<float> tmp(outputLayer.size(), 0.0);

        for (size_t outputIdx{0}; outputIdx < target.size(); ++outputIdx) {
            auto const t{target[outputIdx]};

            // set the correct position, for each input there's a corresponding output
            tmp[t] = 1.0f;

            stagingBuffer.copyData(reinterpret_cast<uint8_t const*>(tmp.data()),  //
                                   singleOutputSizeBytes,  //
                                   singleOutputSizeBytes * outputIdx);

            // reset the position after copy
            tmp[t] = 0.0f;
        }

        graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
                                          m_expectedOutput,  //
                                          std::move(stagingBuffer),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_WRITE_BIT,  //
                                          graphicsManager.computeQueue);
    }

    std::vector<size_t> indices(input.size());  // Indices for shuffling
    std::iota(indices.begin(), indices.end(), 0);

    common::Timer totalTimer{};

    totalTimer.start();

    for (size_t epoch{0}; epoch < epochCount; ++epoch) {
        // float epochLoss{0.0};

        fmt::println("epoch {} {}", epoch, indices.size());

        std::array<VkCommandBuffer, graphics::CONCURRENT_ITERATIONS_COUNT> commandBuffers{};

        {
            VkCommandBufferAllocateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            info.pNext = nullptr;
            info.commandPool = graphicsManager.commandManager.m_commandPool;
            info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount = commandBuffers.size();

            if (vkAllocateCommandBuffers(graphicsManager.device, &info, commandBuffers.data()) != VK_SUCCESS) {
                throw std::runtime_error{"Failed to allocate command buffers."};
            }
        }

        for (size_t i{0}; i < indices.size(); ++i) {
            auto const currFence{m_fences[m_currIteration]};

            common::Timer t1{};
            t1.start();
            if (vkWaitForFences(graphicsManager.device, 1, &currFence, VK_TRUE, std::numeric_limits<uint64_t>::max()) !=
                VK_SUCCESS) {
                throw std::runtime_error{fmt::format("Failed to wait for NeuralNetwork fence {}.", m_currIteration)};
            }

            if (vkResetFences(graphicsManager.device, 1, &currFence) != VK_SUCCESS) {
                throw std::runtime_error{fmt::format("Failed to reset NeuralNetwork fence {}.", m_currIteration)};
            }
            static uint32_t counter{0};
            static double d1{0.0};
            d1 += t1.stop();

            auto const commandBuffer{commandBuffers[m_currIteration]};

            VkCommandBufferBeginInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            info.pNext = nullptr;
            info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            info.pInheritanceInfo = nullptr;

            if (vkBeginCommandBuffer(commandBuffers[m_currIteration], &info) != VK_SUCCESS) {
                throw std::runtime_error{"Failed to begin command buffer."};
            }

            auto const idx{indices[i]};

            // can throw
            common::Timer t2{};
            t2.start();
            forward(graphicsManager, commandBuffer, idx, m_currIteration);
            static double d2{0.0};
            d2 += t2.stop();

            // can throw
            common::Timer t3{};
            t3.start();
            backward(graphicsManager, commandBuffer, idx, learningRate, m_currIteration);
            static double d3{0.0};
            d3 += t3.stop();

            common::Timer t4{};
            t4.start();
            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                throw std::runtime_error{"Failed to end copy command buffer."};
            }

            graphics::submit(commandBuffer, graphicsManager.computeQueue.queue, currFence);

            m_currIteration = (m_currIteration + 1) % graphics::CONCURRENT_ITERATIONS_COUNT;

            static double d4{0.0};
            d4 += t4.stop();

            ++counter;
            // fmt::println("fence {:.5f}, forward {:.5f}, backward {:.5f}, submit {:.5f}", d1 / counter, d2 / counter,
            //              d3 / counter, d4 / counter);
        }

        // vkQueueWaitIdle(graphicsManager.computeQueue.queue);
        // std::vector<float> test(layers[2].size());
        // graphics::utils::read_float_helper(graphicsManager,  //
        //                                    layers[2].values,
        //                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
        //                                    VK_ACCESS_SHADER_WRITE_BIT,  //
        //                                    test);
        // fmt::println("target {}", target[indices[99]]);
        // fmt::println("{}", test);

        shuffle_indices(indices);

        // float const averageLoss{epochLoss / input.size()};
        // fmt::println("Epoch {}:\n\taverage loss: {}\n\tepoch time: {:.2f} ms", epoch, averageLoss,
        // epochTimer.stop());
    }

    auto const totalTimeMs{totalTimer.stop()};
    fmt::println("Training completed in {:.2f} ms", totalTimeMs);
    fmt::println("Average epoch time: {:.2f} ms", totalTimeMs / epochCount);
}

auto NeuralNetwork::forward(graphics::GraphicsManager& graphicsManager,  //
                            VkCommandBuffer commandBuffer,  //
                            uint32_t dataIndex,  //
                            uint32_t iterationIndex  //
                            ) -> void {
    common::Timer t2{};
    t2.start();
    for (size_t i{1}; i < layers.size(); ++i) {
        auto& currLayer{layers[i]};
        auto& prevLayer{layers[i - 1]};

        uint32_t const inputDataIndex{(i == 1) ? dataIndex : 0};

        currLayer.activate(graphicsManager, commandBuffer, prevLayer, iterationIndex, inputDataIndex);
    }
    if (auto const t{t2.stop()}; t > 3.0) {
        fmt::println("forward 2 {:.2f} ms", t);
    }
}

auto NeuralNetwork::backward(graphics::GraphicsManager& graphicsManager,  //
                             VkCommandBuffer commandBuffer,  //
                             uint32_t dataIndex,  //
                             float learningRate,  //
                             uint32_t iterationIndex  //
                             ) -> void {
    auto& outputLayer{layers.back()};

    // calculate deltas
    {
        // special case - output layer
        graphics::calculate_output_delta(graphicsManager,  //
                                         commandBuffer,  //
                                         m_outputDeltaDescriptorSets[iterationIndex],  //
                                         outputLayer.size(),  //
                                         outputLayer.values,  //
                                         m_expectedOutput,  //
                                         outputLayer.delta,  //
                                         outputLayer.size(),  //
                                         dataIndex);

        auto const& dSets{m_hiddenDeltaDescriptorSets[iterationIndex]};

        // the hidden layers
        for (size_t layerInd{layers.size() - 2}; layerInd > 0; --layerInd) {
            auto& layer{layers[layerInd]};
            auto const& rightLayer{layers[layerInd + 1]};

            graphics::calculate_hidden_delta(graphicsManager,  //
                                             commandBuffer,  //
                                             dSets[layerInd],  //
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

            uint32_t const inputDataIndex{(layerInd == 1) ? dataIndex : 0};

            // can throw
            layer.update(graphicsManager, commandBuffer, leftLayer, learningRate, iterationIndex, inputDataIndex);
        }
    }
}

auto NeuralNetwork::clear(graphics::GraphicsManager const& graphicsManager) noexcept -> void {
    for (auto& layer : layers) {
        layer.clear();
    }

    m_expectedOutput.destroy();

    for (size_t i{0}; i < m_fences.size(); ++i) {
        vkDestroyFence(graphicsManager.device, m_fences[i], nullptr);
        m_fences[i] = VK_NULL_HANDLE;
    }
}

}  // namespace impl
