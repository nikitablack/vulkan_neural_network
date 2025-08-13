#pragma once

#include <impl/Layer.hpp>

namespace impl {

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<size_t> const& layerSizes);

    [[nodiscard]] auto forward(std::vector<Float> const& inputValues,  //
                               std::vector<Float>& outputValues  //
                               ) noexcept -> bool;

    [[nodiscard]] auto train(std::vector<std::vector<Float>> const& input,  //
                             std::vector<uint8_t> const& target,  //
                             size_t epochCount,  //
                             Float learningRate  //
                             ) noexcept -> bool;

private:
    [[nodiscard]] auto backward(std::vector<Float> const& output,  //
                                std::vector<Float> const& expectedOutput,  //
                                Float learningRate  //
                                ) noexcept -> bool;

    auto print() const noexcept -> void;

public:
    static auto sigmoid(Float v) noexcept -> Float;
    static auto sigmoidDerivative(Float v) noexcept -> Float;

public:
    std::vector<Layer> layers;
};

}  // namespace impl
