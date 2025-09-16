#include <fmt/core.h>

#include <common/LCG.hpp>
#include <common/Timer.hpp>
#include <impl/NeuralNetwork.hpp>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

auto shuffle_indices(std::vector<size_t>& indices) noexcept -> void {
    std::mt19937 engine{100};
    std::shuffle(indices.begin(), indices.end(), engine);
}

}  // namespace

namespace impl {

NeuralNetwork::NeuralNetwork(std::vector<size_t> const& layerSizes, common::LCG<common::Float>& lcg) {
    if (layerSizes.size() < 2) {
        throw std::runtime_error{"NeuralNetwork must have at least an input and an output layer."};
    }

    layers.emplace_back(layerSizes[0], 0, lcg);  // Input layer

    for (size_t i{1}; i < layerSizes.size(); ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i - 1], lcg);
    }
}

auto NeuralNetwork::forward(std::vector<common::Float> const& inputValues,  //
                            std::vector<common::Float>& outputValues  //
                            ) noexcept -> bool {
    auto& inputLayer{layers.front()};

    if (inputValues.size() != inputLayer.neurons.size()) {
        return false;  // Mismatch in input size
    }

    for (size_t i{0}; i < inputLayer.neurons.size(); ++i) {
        inputLayer.neurons[i].value = inputValues[i];
    }

    for (size_t i{1}; i < layers.size(); ++i) {
        auto& currLayer{layers[i]};
        auto& prevLayer{layers[i - 1]};

        if (!currLayer.activate(prevLayer, sigmoid)) {
            return false;
        }
    }

    auto const& outputLayer{layers.back()};

    outputValues.resize(outputLayer.neurons.size());
    for (size_t i{0}; i < outputLayer.neurons.size(); ++i) {
        outputValues[i] = outputLayer.neurons[i].value;
    }

    return true;
}

auto NeuralNetwork::train(std::vector<std::vector<common::Float>> const& input,  // 0.0-1.0
                          std::vector<uint8_t> const& target,  // 0-9
                          size_t epochCount,  //
                          common::Float learningRate  //
                          ) noexcept -> bool {
    using namespace common;

    if (input.size() != target.size()) {
        fmt::println("Mismatch between input size and target size.");
        return false;
    }

    std::vector<Float> output{};  // buffer for multiple forward passes
    std::vector<Float> expectedOutput{};  // buffer for expected output

    std::vector<size_t> indices(input.size());  // Indices for shuffling
    std::iota(indices.begin(), indices.end(), 0);

    Timer totalTimer{};
    Timer epochTimer{};

    totalTimer.start();

    for (size_t epoch{0}; epoch < epochCount; ++epoch) {
        epochTimer.start();

        Float epochLoss{0.0_F};

        for (size_t i{0}; i < indices.size(); ++i) {
            auto const idx{indices[i]};

            if (!forward(input[idx], output)) {
                fmt::println("Failed to compute forward pass.");
                return false;
            }

            auto const& outputLayer{layers.back()};

            expectedOutput.resize(outputLayer.neurons.size());
            expectedOutput.assign(expectedOutput.size(), 0.0_F);

            if (target[idx] >= expectedOutput.size()) {
                fmt::println("Target index {} out of bounds for output size {}.", target[idx], expectedOutput.size());
                return false;
            }

            expectedOutput[target[idx]] = 1.0_F;

            Float mse{0.0_F};
            for (size_t j{0}; j < output.size(); ++j) {
                Float const diff{expectedOutput[j] - output[j]};
                mse += diff * diff;
            }
            mse /= output.size();  // Mean squared error

            epochLoss += mse;

            if (!backward(output, expectedOutput, learningRate)) {
                return false;  // Backward pass failed
            }
        }

        shuffle_indices(indices);

        Float const averageLoss{epochLoss / input.size()};
        fmt::println("Epoch {}:\n\taverage loss: {}\n\tepoch time: {:.2f} ms", epoch, averageLoss, epochTimer.stop());
    }

    auto const totalTimeMs{totalTimer.stop()};
    fmt::println("Training completed in {:.2f} ms", totalTimeMs);
    fmt::println("Average epoch time: {:.2f} ms", totalTimeMs / epochCount);

    return true;
}

auto NeuralNetwork::backward(std::vector<common::Float> const& output,  //
                             std::vector<common::Float> const& expectedOutput,  //
                             common::Float learningRate  //
                             ) noexcept -> bool {
    using namespace common;

    if (output.size() != expectedOutput.size()) {
        return false;  // Mismatch in output size
    }

    auto& outputLayer{layers.back()};

    if (output.size() != outputLayer.neurons.size()) {
        return false;  // Mismatch in output layer size
    }

    // calculate deltas
    {
        // special case - output layer
        for (size_t i{0}; i < output.size(); ++i) {
            Float const a{output[i]};
            Float const dCdA{2.0_F * (a - expectedOutput[i]) / output.size()};
            Float const dAdZ{sigmoidDerivative(a)};
            outputLayer.delta[i] = dCdA * dAdZ;
        }

        for (size_t layerInd{layers.size() - 2}; layerInd > 0; --layerInd) {
            auto& layer{layers[layerInd]};
            auto const& rightLayer{layers[layerInd + 1]};

            // currLayer.neurons.size == rightLayer.neurons[X].weights.size
            for (size_t i{0}; i < layer.neurons.size(); ++i) {
                Float deltaSum{0.0};

                for (size_t j{0}; j < rightLayer.neurons.size(); ++j) {
                    auto& rightLayerNeuron{rightLayer.neurons[j]};

                    deltaSum += rightLayer.delta[j] * rightLayerNeuron.weights[i];
                }

                layer.delta[i] = deltaSum * sigmoidDerivative(layer.neurons[i].value);
            }
        }
    }

    // update layers
    {
        for (size_t layerInd{1}; layerInd < layers.size(); ++layerInd) {
            auto& layer{layers[layerInd]};
            auto const& leftLayer{layers[layerInd - 1]};

            if (!layer.update(leftLayer, learningRate)) {
                return false;
            }
        }
    }

    return true;
}

auto NeuralNetwork::sigmoid(common::Float v) noexcept -> common::Float {
    using namespace common;

    return 1.0_F / (1.0_F + std::exp(-v));
}

auto NeuralNetwork::sigmoidDerivative(common::Float sigmoidResult) noexcept -> common::Float {
    using namespace common;

    return sigmoidResult * (1.0_F - sigmoidResult);
}

auto NeuralNetwork::print() const noexcept -> void {
    for (size_t k{1}; k < layers.size(); ++k) {
        auto& layer{layers[k]};
        std::cout << "Layer " << k << "\n";
        size_t neuronIndex{0};
        for (const auto& neuron : layer.neurons) {
            std::cout << neuronIndex++ << " Weights: ";
            for (const auto& weight : neuron.weights) {
                std::cout << weight << " ";
            }
            std::cout << "\n";
            std::cout << "  Bias: " << neuron.bias << "\n";
            std::cout << "  Value: " << neuron.value << "\n";
        }
        std::cout << "\n";
    }
}

}  // namespace impl
