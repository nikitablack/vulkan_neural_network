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

[[maybe_unused]] auto print_matrix(impl::Layer::MatrixX const& m) -> void {
    std::cout << "[";
    for (int r{0}; r < m.rows(); ++r) {
        for (int c{0}; c < m.cols(); ++c) {
            if (r == (m.rows() - 1) && c == (m.cols() - 1)) {
                std::cout << m(r, c) << "]";
            } else {
                std::cout << m(r, c) << ", ";
            }
        }
    }
    std::cout << std::endl;
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

    if (inputValues.size() != inputLayer.size()) {
        return false;  // mismatch in input size
    }

    // copy input values to the input layer
    inputLayer.values.col(0).segment(0, inputValues.size()) =
        Eigen::Map<const Layer::VectorX>{inputValues.data(),  //
                                         static_cast<Eigen::Index>(inputValues.size())};

    for (size_t i{1}; i < layers.size(); ++i) {
        auto& currLayer{layers[i]};
        auto& prevLayer{layers[i - 1]};

        if (!currLayer.activate(prevLayer, sigmoid)) {
            return false;
        }
    }

    auto const& outputLayer{layers.back()};

    outputValues.resize(outputLayer.size());
    outputValues.assign(outputLayer.values.col(0).data(),  //
                        outputLayer.values.col(0).data() + outputLayer.size());

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

    std::vector<size_t> indices(input.size());  // Indices for shuffling
    std::iota(indices.begin(), indices.end(), 0);

    Timer totalTimer{};
    Timer epochTimer{};

    totalTimer.start();

    for (size_t epoch{0}; epoch < epochCount; ++epoch) {
        epochTimer.start();

        Float epochLoss{0.0};

        for (size_t i{0}; i < indices.size(); ++i) {
            auto const idx{indices[i]};

            if (!forward(input[idx], output)) {
                fmt::println("Failed to compute forward pass.");
                return false;
            }

            auto const& outputLayer{layers.back()};

            std::vector<Float> expectedOutput(outputLayer.size(), 0.0);
            expectedOutput[target[idx]] = 1.0;

            Float loss{0.0};
            for (size_t j{0}; j < output.size(); ++j) {
                Float const diff{expectedOutput[j] - output[j]};
                loss += diff * diff;
            }
            loss /= output.size();  // Mean squared error

            epochLoss += loss;

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

auto NeuralNetwork::print() const noexcept -> void {
    for (size_t i{1}; i < layers.size(); ++i) {
        auto const& layer{layers[i]};
        std::cout << "Layer " << i << "\n";
        std::cout << "Weights:\n" << layer.weights << "\n";
        std::cout << "Biases:\n" << layer.biases << "\n";
        std::cout << "Values:\n" << layer.values << "\n";
    }
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

    if (output.size() != outputLayer.size()) {
        return false;  // Mismatch in output layer size
    }

    // calculate deltas
    {
        // special case - output layer
        for (size_t i{0}; i < output.size(); ++i) {
            Float const a{output[i]};
            Float const dCdA{2.0_F * (a - expectedOutput[i]) / output.size()};
            Float const dAdZ{sigmoidDerivative(a)};
            outputLayer.delta(static_cast<Eigen::Index>(i), 0) = dCdA * dAdZ;
        }

        for (size_t layerInd{layers.size() - 2}; layerInd > 0; --layerInd) {
            auto& layer{layers[layerInd]};
            auto const& rightLayer{layers[layerInd + 1]};

            Layer::MatrixX const WT{rightLayer.weights.transpose()};
            Layer::MatrixX WTD{WT * rightLayer.delta};
            Layer::MatrixX dZ{layer.values.unaryExpr(&sigmoidDerivative)};
            layer.delta = WTD.cwiseProduct(dZ);
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

}  // namespace impl
