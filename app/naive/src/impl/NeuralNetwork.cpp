#include <fmt/core.h>

#include <impl/NeuralNetwork.hpp>
#include <impl/Timer.hpp>
#include <iostream>
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

NeuralNetwork::NeuralNetwork(std::vector<size_t> const& layerSizes) {
    if (layerSizes.size() < 2) {
        throw std::runtime_error{"NeuralNetwork must have at least an input and an output layer."};
    }

    layers.emplace_back(layerSizes[0], 0);  // Input layer

    for (size_t i{1}; i < layerSizes.size(); ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i - 1]);
    }
}

auto NeuralNetwork::forward(std::vector<Float> const& inputValues,  //
                            std::vector<Float>& outputValues  //
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

auto NeuralNetwork::train(std::vector<std::vector<Float>> const& input,  // 0.0-1.0
                          std::vector<uint8_t> const& target,  // 0-9
                          size_t epochCount,  //
                          Float learningRate  //
                          ) noexcept -> bool {
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

auto NeuralNetwork::backward(std::vector<Float> const& output,  //
                             std::vector<Float> const& expectedOutput,  //
                             Float learningRate  //
                             ) noexcept -> bool {
    if (output.size() != expectedOutput.size()) {
        return false;  // Mismatch in output size
    }

    auto& outputLayer{layers.back()};

    if (output.size() != outputLayer.neurons.size()) {
        return false;  // Mismatch in output layer size
    }

    // update output layer
    std::vector<Float> deltaOutput(output.size());
    for (size_t i{0}; i < output.size(); ++i) {
        Float const a{output[i]};
        Float const dCdA{2.0_F * (a - expectedOutput[i]) / output.size()};
        Float const dAdZ{sigmoidDerivative(a)};
        deltaOutput[i] = dCdA * dAdZ;
    }

    {
        auto const& leftLayer{layers[layers.size() - 2]};

        if (!outputLayer.update(leftLayer, learningRate, deltaOutput)) {
            return false;
        }
    }

    // update hidden layers
    {
        auto delta{std::move(deltaOutput)};

        auto const* rightLayer{&outputLayer};

        for (size_t lay{layers.size() - 2}; lay > 0; --lay) {
            auto& currLayer{layers[lay]};
            auto const& leftLayer{layers[lay - 1]};

            std::vector<Float> deltaHidden(currLayer.neurons.size());

            // delta.size == rightLayer.neurons.size
            // currLayer.neurons.size == rightLayer.neurons[X].weights.size
            for (size_t i{0}; i < currLayer.neurons.size(); ++i) {
                Float deltaSum{0.0};

                for (size_t j{0}; j < rightLayer->neurons.size(); ++j) {
                    auto& neuron{rightLayer->neurons[j]};

                    deltaSum += delta[j] * neuron.weights[i];
                }

                deltaHidden[i] = deltaSum * sigmoidDerivative(currLayer.neurons[i].value);
            }

            if (!currLayer.update(leftLayer, learningRate, deltaHidden)) {
                return false;
            }

            // Prepare for the next iteration
            delta = std::move(deltaHidden);
            rightLayer = &currLayer;
        }
    }

    return true;
}

auto NeuralNetwork::sigmoid(Float v) noexcept -> Float {
    return 1.0_F / (1.0_F + std::exp(-v));
}

auto NeuralNetwork::sigmoidDerivative(Float sigmoidResult) noexcept -> Float {
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
