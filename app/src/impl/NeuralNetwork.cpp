#include <fmt/core.h>

#include <impl/NeuralNetwork.hpp>
#include <impl/ScopedTimer.hpp>
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

auto NeuralNetwork::forward(std::vector<double> const& inputValues,  //
                            std::vector<double>& outputValues  //
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

auto NeuralNetwork::train(std::vector<std::vector<double>> const& input,  // 0.0-1.0
                          std::vector<uint8_t> const& target,  // 0-9
                          size_t epochCount,  //
                          double learningRate  //
                          ) noexcept -> bool {
    if (input.size() != target.size()) {
        fmt::println("Mismatch between input size and target size.");
        return false;
    }

    std::vector<double> output{};  // buffer for multiple forward passes

    std::vector<size_t> indices(input.size());  // Indices for shuffling
    std::iota(indices.begin(), indices.end(), 0);

    ScopedTimer t1{fmt::format("NeuralNetwork train time: ")};
    for (size_t epoch{0}; epoch < epochCount; ++epoch) {
        ScopedTimer t2{fmt::format("Epoch {}: ", epoch)};

        double epochLoss{0.0};

        for (size_t i{0}; i < indices.size(); ++i) {
            auto const idx{indices[i]};

            if (!forward(input[idx], output)) {
                fmt::println("Failed to compute forward pass.");
                return false;
            }

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {
                std::cout << "Before " << epoch << "\n";

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
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            auto const& outputLayer{layers.back()};

            std::vector<double> expectedOutput(outputLayer.neurons.size(), 0.0);
            expectedOutput[target[idx]] = 1.0;

            double loss{0.0};
            for (size_t j{0}; j < output.size(); ++j) {
                double const diff{expectedOutput[j] - output[j]};
                loss += diff * diff;
            }
            loss /= output.size();  // Mean squared error

            epochLoss += loss;

            if (!backward(output, expectedOutput, learningRate)) {
                return false;  // Backward pass failed
            }

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {
                std::cout << "After " << epoch << "\n";

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
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        }

        shuffle_indices(indices);

        double const averageLoss{epochLoss / input.size()};
        fmt::println("Epoch {} average loss: {}", epoch, averageLoss);
    }

    return true;
}

auto NeuralNetwork::backward(std::vector<double> const& output,  //
                             std::vector<double> const& expectedOutput,  //
                             double learningRate  //
                             ) noexcept -> bool {
    if (output.size() != expectedOutput.size()) {
        return false;  // Mismatch in output size
    }

    auto& outputLayer{layers.back()};

    if (output.size() != outputLayer.neurons.size()) {
        return false;  // Mismatch in output layer size
    }

    // update output layer
    std::vector<double> deltaOutput(output.size());
    for (size_t i{0}; i < output.size(); ++i) {
        double const a{output[i]};
        double const dCdA{2.0 * (a - expectedOutput[i]) / output.size()};
        double const dAdZ{sigmoidDerivative(a)};
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

            std::vector<double> deltaHidden(rightLayer->inputCount);

            // delta.size == rightLayer.neurons.size
            // currLayer.neurons.size == rightLayer.neurons[X].weights.size
            for (size_t i{0}; i < currLayer.neurons.size(); ++i) {
                double deltaSum{0.0};

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

auto NeuralNetwork::sigmoid(double v) noexcept -> double {
    return 1.0 / (1.0 + std::exp(-v));
}

auto NeuralNetwork::sigmoidDerivative(double sigmoidResult) noexcept -> double {
    return sigmoidResult * (1.0 - sigmoidResult);
}

}  // namespace impl
