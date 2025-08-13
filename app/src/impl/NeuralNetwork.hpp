#pragma once

#include <impl/Layer.hpp>

namespace impl {

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<size_t> const& layerSizes);

    [[nodiscard]] auto forward(std::vector<double> const& inputValues,  //
                               std::vector<double>& outputValues  //
                               ) noexcept -> bool;

    [[nodiscard]] auto train(std::vector<std::vector<double>> const& input,  //
                             std::vector<uint8_t> const& target,  //
                             size_t epochCount,  //
                             double learningRate  //
                             ) noexcept -> bool;

private:
    [[nodiscard]] auto backward(std::vector<double> const& output,  //
                                std::vector<double> const& expectedOutput,  //
                                double learningRate  //
                                ) noexcept -> bool;

    auto print() const noexcept -> void;

public:
    static auto sigmoid(double v) noexcept -> double;
    static auto sigmoidDerivative(double v) noexcept -> double;

public:
    std::vector<Layer> layers;
};

}  // namespace impl
